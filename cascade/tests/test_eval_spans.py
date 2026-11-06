import unittest

from cascade.eval_proposals import annotated_spans, match


class AnnotatedSpansTest(unittest.TestCase):
    def test_long_gap_splits_and_short_gap_keeps(self):
        fps = 120.0
        # 0~10 s 一段，14 s 处死球后再来一段（间隔 4 s，不切），之后空档 100 s
        markup = {120: "bounce", 1200: "net", 1680: "bounce", 1680 + 12000: "bounce"}
        spans = annotated_spans(markup, fps)
        self.assertEqual(spans, [[0, 1800], [13560, 13800]])

    def test_outside_predictions_dropped_from_fp(self):
        markup = {100: "bounce", 200: "bounce", 100000: "bounce"}
        spans = annotated_spans(markup, 120.0)
        pred = [101, 199, 50000, 50300]           # 空档里的两条不应算误检
        inside = [f for f in pred if any(a <= f <= b for a, b in spans)]
        r = match(inside, sorted(f for f, v in markup.items() if v == "bounce"), 5)
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (2, 0, 1))


if __name__ == "__main__":
    unittest.main()
