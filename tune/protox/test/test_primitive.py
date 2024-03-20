import unittest
from tune.protox.env.space.primitive.knob import Knob
from tune.protox.env.space.primitive.latent_knob import LatentKnob
from tune.protox.env.space.primitive.index import IndexAction

class PrimitivesTests(unittest.TestCase):

    def test_linear_knob(self):
        k = Knob(
            table_name=None,
            query_name="q",
            knob_name="kn",
            metadata={
                "type": "float",
                "min": 0.,
                "max": 1.,
                "quantize": 10,
                "log_scale": False,
                "unit": 0,
            },
            do_quantize=True,
            default_quantize_factor=10,
            seed=0,
        )
        self.assertEqual(k.name(), "q_kn")
        self.assertEqual(k.bucket_size, 0.1)
        self.assertEqual(k.project_scraped_setting(0.5), 0.5)
        self.assertEqual(k.project_scraped_setting(0.58), 0.5)
        self.assertEqual(round(k.project_scraped_setting(0.62), 2), 0.6)

    def test_log_knob(self):
        k = Knob(
            table_name=None,
            query_name="q",
            knob_name="kn",
            metadata={
                "type": "integer",
                "min": 1.,
                "max": 1024.,
                "quantize": 0,
                "log_scale": True,
                "unit": 0,
            },
            do_quantize=True,
            default_quantize_factor=10,
            seed=0,
        )
        self.assertEqual(k.name(), "q_kn")
        self.assertEqual(k.project_scraped_setting(1), 1.0)
        self.assertEqual(k.project_scraped_setting(2), 2.0)
        self.assertEqual(k.project_scraped_setting(24), 32.0)
        self.assertEqual(k.project_scraped_setting(1024), 1024.0)

    def test_latent_knob(self):
        k = LatentKnob(
            table_name=None,
            query_name="q",
            knob_name="kn",
            metadata={
                "type": "float",
                "min": 0.,
                "max": 1.,
                "quantize": 10,
                "log_scale": False,
                "unit": 0,
            },
            do_quantize=True,
            default_quantize_factor=10,
            seed=0,
        )
        self.assertEqual(k.name(), "q_kn")
        self.assertEqual(k.bucket_size, 0.1)
        self.assertEqual(k.project_scraped_setting(0.5), 0.5)
        self.assertEqual(k.project_scraped_setting(0.58), 0.5)
        self.assertEqual(round(k.project_scraped_setting(0.62), 2), 0.6)

        self.assertEqual(k.to_latent(0.5), 0.0)
        self.assertEqual(k.from_latent(-1.), 0.0)
        self.assertEqual(k.from_latent(1.), 1.0)
        self.assertEqual(k.from_latent(0.5), 0.7)

        self.assertEqual(k.shift_offset(0.5, 0), None)
        self.assertEqual(k.shift_offset(0.5, 1), 0.6)
        self.assertEqual(k.shift_offset(0.5, -2), 0.3)

    def test_ia(self):
        ia1 = IndexAction(
            idx_type="btree",
            tbl="tbl",
            columns=["a", "b", "c"],
            col_idxs=None,
            inc_names=["d", "e"],
            raw_repr=None,
            bias=0.0,
        )
        IndexAction.index_counter = 0
        self.assertEqual(ia1.sql(add=True), "CREATE INDEX  index1 ON tbl USING btree (a,b,c) INCLUDE (d,e)")

        ia2 = IndexAction(
            idx_type="btree",
            tbl="tbl",
            columns=["a", "b", "c"],
            col_idxs=None,
            inc_names=["d", "e"],
            raw_repr=None,
            bias=0.0,
        )
        self.assertEqual(ia1, ia2)

        ia3 = IndexAction(
            idx_type="btree",
            tbl="tbl",
            columns=["a", "b", "c"],
            col_idxs=None,
            inc_names=[],
            raw_repr=None,
            bias=0.0,
        )
        self.assertNotEqual(ia1, ia3)

        ia4 = IndexAction(
            idx_type="btree",
            tbl="tbl",
            columns=["a", "b"],
            col_idxs=None,
            inc_names=["d", "e"],
            raw_repr=None,
            bias=0.0,
        )
        self.assertNotEqual(ia1, ia4)

        ia5 = IndexAction(
            idx_type="btree",
            tbl="tbla",
            columns=["a", "b", "c"],
            col_idxs=None,
            inc_names=["d", "e"],
            raw_repr=None,
            bias=0.0,
        )
        self.assertNotEqual(ia1, ia5)
