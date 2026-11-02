from model import BallTrackerNet
import torch
from datasets import trackNetDataset
import torch.optim as optim
import os
from tensorboardX import SummaryWriter
from general import train, validate
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--exp_id', type=str, default='default', help='path to saving results')
    parser.add_argument('--num_epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation')
    parser.add_argument('--steps_per_epoch', type=int, default=200, help='number of steps per one epoch')
    parser.add_argument('--init_weights', type=str, default=None,
                        help='optional path to pretrained weights to start from '
                             '(use the tennis model_best.pt for transfer learning)')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to model_last.pt to resume an interrupted run '
                             '(overrides --init_weights; pair with --start_epoch)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='first epoch index when resuming (0 = fresh run)')
    parser.add_argument('--best_f1', type=float, default=0.0,
                        help='initial best val f1 when resuming, so resumed epochs '
                             'cannot overwrite a better earlier model_best.pt')
    parser.add_argument('--dataset_path', type=str, default='./datasets/trackNet',
                        help='root of the prepared dataset')
    parser.add_argument('--input_height', type=int, default=360)
    parser.add_argument('--input_width', type=int, default=640)
    parser.add_argument('--min_dist', type=int, default=5,
                        help='hit radius for the val metric (tennis=5, table tennis ~8-10)')
    parser.add_argument('--min_radius', type=int, default=2,
                        help='HoughCircles min radius (tennis=2)')
    parser.add_argument('--max_radius', type=int, default=7,
                        help='HoughCircles max radius (tennis=7, table tennis ~12)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (original hard-coded 1; raise for speed)')
    parser.add_argument('--amp', action='store_true',
                        help='use fp16 autocast for training (faster, ~half VRAM, '
                             'lets you raise batch_size)')
    parser.add_argument('--augment', action='store_true',
                        help='enable on-the-fly augmentation on the train split '
                             '(colour/brightness/scale/translate/rotate/flip). '
                             'Recommended when training on a single game to close the '
                             'domain gap to other venues.')
    parser.add_argument('--optim', type=str, default='adadelta',
                        choices=['adadelta', 'adamw', 'adam', 'sgd'],
                        help='optimizer. TrackNet uses adadelta lr=1.0; when switching '
                             'to adamw/adam use a much smaller lr (1e-4 ~ 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 regularisation (adadelta: try 1e-4 to fight over-fitting)')
    args = parser.parse_args()
    
    train_dataset = trackNetDataset('train', args.input_height, args.input_width,
                                    args.dataset_path, augment=args.augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    
    val_dataset = trackNetDataset('val', args.input_height, args.input_width, args.dataset_path)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )    
    
    model = BallTrackerNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print('resumed from {} (start_epoch = {})'.format(args.resume, args.start_epoch))
    elif args.init_weights:
        state = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(state)
        print('initialised from {}'.format(args.init_weights))
    
    exps_path = './exps/{}'.format(args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    if args.optim == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)
    print('optimizer = {} | lr = {} | weight_decay = {}'.format(
        args.optim, args.lr, args.weight_decay), flush=True)
    val_best_metric = args.best_f1

    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss = train(model, train_loader, optimizer, device, epoch, args.steps_per_epoch,
                           amp=args.amp)
        print('train loss = {}'.format(train_loss), flush=True)
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)
        log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_intervals == 0:
            val_loss, precision, recall, f1 = validate(
                model, val_loader, device, epoch, min_dist=args.min_dist,
                out_height=args.input_height, out_width=args.input_width,
                min_radius=args.min_radius, max_radius=args.max_radius,
                verbose=False, amp=args.amp)
            print('val loss = {} | precision = {:.4f} | recall = {:.4f} | f1 = {:.4f}'.format(
                val_loss, precision, recall, f1), flush=True)
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/recall', recall, epoch)
            log_writer.add_scalar('Val/f1', f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                torch.save(model.state_dict(), model_best_path)
                print('  new best f1 -> saved model_best.pt', flush=True)
        # always checkpoint so an interrupted run still leaves usable weights
        torch.save(model.state_dict(), model_last_path)
        log_writer.flush()
