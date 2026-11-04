<#
.SYNOPSIS
  第 5 步一键脚本：在原机（有 GPU 和 OpenTTGames 测试视频）上产出级联需要的全部缓存，
  再用 replay 模式跑级联与评估。

.DESCRIPTION
  依次执行：
    0. 环境自检（python / torch / CUDA / easydict / 各路径）
    1. TrackNet 适配器：test_1、test_3 ~ test_7 → _data\obs\tracknet\<game>.jsonl
       （test_2 的 TrackNet 观测仓库里已有；加 -IncludeTest2TrackNet 可重跑）
       新机位没有球台四角时，导出带坐标网格的帧到 _data\calib\<game>_grid.jpg，
       需要你人工读四角写进 calib\table_quads.json（见 README「新机位标定」）。
    2. YOLO 导出：test_2 → _data\obs\yolo\test_2.jsonl
    3. TTNet 逐窗口缓存：test_2 → _data\ttnet\test_2.jsonl（+ .meta.json，含单窗耗时）
    4. replay：run_cascade（--reviewer ttnet）、eval_proposals、eval_extended
       → outputs\cascade_step5\
  任何一步失败都会停下并打印原因；已存在的输出默认跳过（-Force 重跑）。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File cascade\tools\run_machine_a.ps1 `
      -TNRoot E:\20-项目\ppq\TN -VideoDir D:\OpenTTGames\test\videos `
      -TTNetWeights D:\assets\ttnet.pth -TTNetSrc E:\20-项目\ppq\TTNet\src
#>
param(
    [Parameter(Mandatory = $true)][string]$TNRoot,
    [Parameter(Mandatory = $true)][string]$VideoDir,
    [Parameter(Mandatory = $true)][string]$TTNetWeights,
    [string]$TTNetSrc = "",                 # 默认 <TNRoot>\..\TTNet\src
    [string]$YoloRepo = "",                 # 默认 <TNRoot>\..\pingpong-smart-referee
    [string]$TrackNetWeights = "exps\tt_g345_aug\model_best.pt",  # 相对 TrackNet\TrackNet
    [string]$Python = "python",             # TN 的 venv：例如 E:\...\TN\.venv\Scripts\python.exe
    [string]$TTNetPython = "",              # TTNet 需要独立 venv 时指定；默认同 $Python
    [string]$Device = "cuda:0",
    [int]$TTNetK = 4,
    [string[]]$TrackNetGames = @("test_1", "test_3", "test_4", "test_5", "test_6", "test_7"),
    [switch]$IncludeTest2TrackNet,
    [switch]$SkipTrackNet,
    [switch]$SkipYolo,
    [switch]$SkipTTNet,
    [switch]$SkipEval,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$env:PYTHONIOENCODING = "utf-8"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Run($exe, [string[]]$argv, $cwd) {
    Write-Host "> $exe $($argv -join ' ')" -ForegroundColor DarkGray
    Push-Location $cwd
    try {
        & $exe @argv
        if ($LASTEXITCODE -ne 0) { throw "命令失败（exit $LASTEXITCODE）：$exe $($argv -join ' ')" }
    } finally { Pop-Location }
}
function NeedRun($path) { return $Force -or -not (Test-Path $path) }

$TNRoot = (Resolve-Path $TNRoot).Path
$PpqRoot = Split-Path $TNRoot -Parent
if (-not $TTNetSrc) { $TTNetSrc = Join-Path $PpqRoot "TTNet\src" }
if (-not $YoloRepo) { $YoloRepo = Join-Path $PpqRoot "pingpong-smart-referee" }
if (-not $TTNetPython) { $TTNetPython = $Python }
$TrackNetDir = Join-Path $TNRoot "TrackNet\TrackNet"
$Calib = Join-Path $TNRoot "calib\table_quads.json"
$ObsTN = Join-Path $TNRoot "_data\obs\tracknet"
$ObsYolo = Join-Path $TNRoot "_data\obs\yolo"
$TTNetOut = Join-Path $TNRoot "_data\ttnet"
$CalibOut = Join-Path $TNRoot "_data\calib"
$OutDir = Join-Path $TNRoot "outputs\cascade_step5"
New-Item -ItemType Directory -Force -Path $ObsTN, $ObsYolo, $TTNetOut, $CalibOut, $OutDir | Out-Null
$Log = Join-Path $OutDir ("run_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $Log | Out-Null

try {
    # ------------------------------------------------------------------ 0
    Step "0. 环境自检"
    foreach ($p in @($VideoDir, $TTNetWeights, $TTNetSrc, $TrackNetDir, $YoloRepo, $Calib)) {
        if (-not (Test-Path $p)) { throw "路径不存在：$p" }
    }
    Run $Python @("-c", "import torch,cv2,numpy;print('torch',torch.__version__,'cuda',torch.cuda.is_available())") $TNRoot
    Run $TTNetPython @("-c", "import torch,easydict;print('ttnet env ok, cuda',torch.cuda.is_available())") $TNRoot
    $sha = (Get-FileHash $TTNetWeights -Algorithm SHA256).Hash.ToLower()
    Write-Host "TTNet 权重 sha256 = $sha"
    if ($sha -eq "370f1b667ee7717fcdb851df88801533665275abacd5fe9df2dba31a0d4ef6d9") {
        Write-Host "  （= SLAM-TT v1.0.0 备用权重 ttnet.pth）"
    } else {
        Write-Host "  （不是 SLAM-TT 备用权重；若是 chagoup 资产包里的权重，请在 README 记下这个 sha256）"
    }
    $calibJson = Get-Content $Calib -Raw -Encoding UTF8 | ConvertFrom-Json

    # ------------------------------------------------------------------ 1
    if (-not $SkipTrackNet) {
        Step "1. TrackNet 适配器出观测"
        $games = @($TrackNetGames)
        if ($IncludeTest2TrackNet) { $games = @("test_2") + $games }
        $missingCalib = @()
        foreach ($g in $games) {
            $video = Join-Path $VideoDir "$g.mp4"
            if (-not (Test-Path $video)) { Warn "缺视频，跳过：$video"; continue }
            $out = Join-Path $ObsTN "$g.jsonl"
            if (NeedRun $out) {
                # tracknet_adapter 要在 TrackNet\TrackNet 下运行（相对权重路径与模型代码）
                Run $Python @("..\..\adapter\tracknet_adapter.py", "--video", $video,
                    "--model_path", $TrackNetWeights, "--out", $out, "--min_conf", "0.5") $TrackNetDir
                Run $Python @("adapter\check_observations.py", $out) $TNRoot
            } else { Write-Host "已存在，跳过：$out" }
            if (-not ($calibJson.PSObject.Properties.Name -contains $g)) {
                $jpg = Join-Path $CalibOut "${g}_grid.jpg"
                Run $Python @("calibrate_table.py", "export", "--video", $video, "--frame", "1200", "--out", $jpg) $TrackNetDir
                $missingCalib += $g
            }
        }
        if ($missingCalib.Count -gt 0) {
            Warn ("这些机位还没有球台四角：" + ($missingCalib -join ", "))
            Warn "已导出带坐标网格的帧到 $CalibOut\<game>_grid.jpg。按 far_left, far_right, near_right, near_left"
            Warn "顺序读出原图坐标，写进 calib\table_quads.json（格式照 test_2），再用"
            Warn "  python calibrate_table.py draw --video <mp4> --frame 1200 --out <jpg> --pts ""x1,y1 x2,y2 x3,y3 x4,y4"""
            Warn "画回去核对（在 TrackNet\TrackNet 下运行；--pts 格式以 calibrate_table.py --help 为准）。"
        }
    }

    # ------------------------------------------------------------------ 2
    if (-not $SkipYolo) {
        Step "2. YOLO 导出 test_2"
        $video = Join-Path $VideoDir "test_2.mp4"
        $out = Join-Path $ObsYolo "test_2.jsonl"
        if (NeedRun $out) {
            Run $Python @("run_referee.py", "--video", $video, "--preset", "test2", "--export-jsonl", $out) $YoloRepo
            Run $Python @("adapter\check_observations.py", $out) $TNRoot
        } else { Write-Host "已存在，跳过：$out" }
    }

    # ------------------------------------------------------------------ 3
    if (-not $SkipTTNet) {
        Step "3. TTNet 逐窗口缓存 test_2"
        $video = Join-Path $VideoDir "test_2.mp4"
        $out = Join-Path $TTNetOut "test_2.jsonl"
        if (NeedRun $out) {
            Run $TTNetPython @("-m", "cascade.ttnet_cache", "--video", $video, "--weights", $TTNetWeights,
                "--ttnet-src", $TTNetSrc, "--engine", "torch", "--device", $Device, "--out", $out) $TNRoot
        } else { Write-Host "已存在，跳过：$out" }
        Get-Content "$out.meta.json" -Encoding UTF8 | Write-Host
        # torch 与 numpy 引擎一致性（需要权重；numpy 一窗约 0.5 s，只跑 20 帧）
        $env:TTNET_WEIGHTS = $TTNetWeights
        $env:TTNET_SRC = $TTNetSrc
        Run $TTNetPython @("-m", "unittest", "cascade.tests.test_ttnet_weights", "-v") $TNRoot
    }

    # ------------------------------------------------------------------ 4
    if (-not $SkipEval) {
        Step "4. replay：k 核实 / run_cascade / eval_proposals / eval_extended"
        $ttCache = Join-Path $TTNetOut "test_2.jsonl"
        $tnObs = Join-Path $ObsTN "test_2.jsonl"
        if (-not (Test-Path $tnObs)) { $tnObs = Join-Path $TNRoot "adapter\examples\test_2_observations.jsonl" }
        $yoloObs = Join-Path $ObsYolo "test_2.jsonl"
        $gtMarkup = Join-Path $TNRoot "_data\gt\extended_openttgames\inferred\test_2_events_markup_inferred.json"
        $gameJson = Join-Path $TNRoot "_data\gt\extended_openttgames\game_data\test\test_2.json"

        # k 核实：同一缓存按 k=0..8 找驼峰，与（推断）真值落台比；标注只到 26.88 s
        Run $Python @("-m", "cascade.ttnet_k_check", "--ttnet-cache", $ttCache, "--gt", $gtMarkup,
            "--game", "test_2", "--gt-end-frame", "3226",
            "--out", (Join-Path $OutDir "k_check_test_2.json")) $TNRoot

        $yoloArg = @()
        if (Test-Path $yoloObs) { $yoloArg = @("--yolo", $yoloObs) } else { Warn "没有 YOLO 观测，run_cascade 不开分歧触发器" }
        $runs = @(
            @{ name = "test_2_rule";  extra = @("--reviewer", "rule") },
            @{ name = "test_2_ttnet_k$TTNetK"; extra = @("--reviewer", "ttnet", "--ttnet-cache", $ttCache, "--ttnet-k", "$TTNetK") }
        )
        foreach ($r in $runs) {
            $rd = Join-Path $OutDir $r.name
            Run $Python (@("-m", "cascade.run_cascade", "--jsonl", $tnObs, "--preset", "test2", "--out-dir", $rd) + $yoloArg + $r.extra) $TNRoot
            Run $Python @("-m", "cascade.eval_proposals", "--run-dir", $rd, "--events-markup", $gtMarkup, "--jsonl", $tnObs) $TNRoot
            Run $Python @("-m", "cascade.eval_extended", "--run-dir", $rd, "--game-json", $gameJson) $TNRoot
        }
        Write-Host "k 核实结果：$(Join-Path $OutDir 'k_check_test_2.json')。若 k_estimate 不是 $TTNetK，用 -TTNetK <新值> -SkipTrackNet -SkipYolo -SkipTTNet 重跑本步。"
    }
    Step "完成。输出在 $OutDir，日志 $Log"
}
finally {
    Stop-Transcript | Out-Null
}
