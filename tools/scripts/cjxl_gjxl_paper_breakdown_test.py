#!/usr/bin/env python3
"""Regression checks for paper regrouping and equal-image attribution."""
import unittest

import pandas as pd

import cjxl_gjxl_paper_breakdown as paper


ALL_PANELS = ("kodak_0_4mp", "clic_1_8_to_3_4mp", "12mp", "24mp", "48mp")


def fixture(resolutions=paper.DEFAULT_PANELS):
    flat, gpu = [], []
    for resolution in resolutions:
        for image, repetitions, host_ms, gpu_ms in (("a", 1, 10, 10), ("b", 3, 30, 0)):
            for sample in range(repetitions):
                key = dict(resolution_class=resolution, image_id=image, setting="Q80",
                           effort=5, sample_index=sample)
                flat.extend([dict(key, kind="flat", stage="Input color transform", ms=host_ms),
                             dict(key, kind="flat", stage="GPU: Other GPU stages", ms=gpu_ms)])
                if gpu_ms:
                    gpu.append(dict(key, stage="aq.gaborish", ms=gpu_ms))
    manifest = dict(efforts=[5], samples=6, cpu_threads=8, source_revision="fixture",
                    images=[dict(resolution_class=r, image_id=i, width=10, height=10)
                            for r in resolutions for i in ("a", "b")])
    return dict(manifest=manifest, samples=pd.DataFrame(flat),
                gpu_stages=pd.DataFrame(gpu), coverage=pd.DataFrame([
                    dict(kind="flat", resolution_class=r, effort=5, status="complete")
                    for r in resolutions]))


class PaperBreakdownTest(unittest.TestCase):
    def tearDown(self):
        paper.plt.close("all")

    def test_absent_stage_is_zero_before_equal_image_average(self):
        samples, means, _ = paper.paper_data(fixture())
        self.assertTrue(samples.groupby(paper.KEYS).size().eq(9).all())
        for resolution in paper.DEFAULT_PANELS:
            part = means[means.resolution_class == resolution].set_index("group")
            self.assertAlmostEqual(part.loc["reconstruct", "ms"], 5)
            self.assertAlmostEqual(part.loc["reconstruct", "percent"], 20)
            self.assertAlmostEqual(part.ms.sum(), 25)

    def test_changed_complete_call_total_is_rejected(self):
        report = fixture()
        report["gpu_stages"].loc[0, "ms"] += 1
        with self.assertRaisesRegex(ValueError, "complete-call partition"):
            paper.paper_data(report)

    def test_incomplete_panel_is_rejected(self):
        report = fixture()
        report["coverage"].loc[0, "status"] = "incomplete"
        with self.assertRaisesRegex(ValueError, "complete image/effort coverage"):
            paper.paper_data(report)

    def test_unknown_stage_is_rejected(self):
        for classifier in (paper.gpu_group, paper.host_group):
            with self.assertRaisesRegex(ValueError, "Unmapped"):
                classifier("unknown.future_stage")

    def test_cli_defaults_and_multiple_panels(self):
        parser = paper.argument_parser()
        required = ["--config", "config.json", "--output-dir", "out"]
        args = parser.parse_args(required)
        self.assertEqual(tuple(args.panels), ("12mp",))
        self.assertFalse(args.show_panel_titles)
        self.assertEqual(args.legend_position, "bottom")
        args = parser.parse_args(required + ["--panels", "kodak", "48mp", "--show-panel-titles",
                                            "--legend-position", "right"])
        self.assertEqual(args.panels, ["kodak", "48mp"])
        self.assertTrue(args.show_panel_titles)
        self.assertEqual(args.legend_position, "right")

    def test_selection_aliases_order_and_errors(self):
        manifest = fixture(ALL_PANELS)["manifest"]
        self.assertEqual(paper.resolve_panels(manifest, ["Unsplash48MP", "Kodak", "CLIC"]),
                         ("48mp", "kodak_0_4mp", "clic_1_8_to_3_4mp"))
        self.assertEqual(paper.resolve_panels(manifest, "12MP"), ("12mp",))
        for selection in ([], ["unknown"], ["kodak", "kodak_0_4mp"]):
            with self.assertRaises(ValueError):
                paper.resolve_panels(manifest, selection)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            paper.resolve_panels(fixture(("48mp",))["manifest"])

    def test_only_selected_data_enter_plot_and_caption(self):
        report = fixture(ALL_PANELS)
        selection = ["48mp", "kodak"]
        samples, means, _ = paper.paper_data(report, selection)
        self.assertEqual(set(samples.resolution_class), {"48mp", "kodak_0_4mp"})
        self.assertEqual(set(means.resolution_class), {"48mp", "kodak_0_4mp"})
        caption = paper.make_caption(report["manifest"], selection)
        self.assertLess(caption.index("Unsplash, 48 MP"), caption.index("Kodak"))
        self.assertNotIn("CLIC", caption)
        self.assertNotIn("12 MP", caption)

    def test_default_single_panel_has_no_headings_and_reclaims_space(self):
        report = fixture(ALL_PANELS)
        _, means, _ = paper.paper_data(report)
        fig = paper.make_figure(means, report["manifest"])
        self.assertEqual([ax.get_label() for ax in fig.axes], ["12mp"])
        self.assertEqual(fig.texts, [])
        titled = paper.make_figure(means, report["manifest"], show_panel_titles=True)
        self.assertEqual(titled.texts[0].get_text(), "(a)  Unsplash, 12 MP")
        self.assertIn("2 images", titled.texts[1].get_text())
        self.assertLess(fig.get_size_inches()[1], titled.get_size_inches()[1])

    def test_ordered_multiple_panels_and_row_wrapping(self):
        report = fixture(ALL_PANELS)
        selection = ["48mp", "kodak", "clic"]
        _, means, _ = paper.paper_data(report, selection)
        fig = paper.make_figure(means, report["manifest"], selection, show_panel_titles=True)
        self.assertEqual([ax.get_label() for ax in fig.axes],
                         ["48mp", "kodak_0_4mp", "clic_1_8_to_3_4mp"])
        self.assertEqual([text.get_text() for text in fig.texts[::2]],
                         ["(a)  Unsplash, 48 MP", "(b)  Kodak", "(c)  CLIC test"])
        self.assertLess(fig.axes[2].get_position().y1, fig.axes[0].get_position().y0)


if __name__ == "__main__":
    unittest.main()
