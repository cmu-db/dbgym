ALTER TABLE nation ADD CONSTRAINT nation_n_regionkey_fkey FOREIGN KEY (n_regionkey) REFERENCES region (r_regionkey) ON DELETE CASCADE;
ALTER TABLE supplier ADD CONSTRAINT supplier_s_nationkey_fkey FOREIGN KEY (s_nationkey) REFERENCES nation (n_nationkey) ON DELETE CASCADE;
ALTER TABLE partsupp ADD CONSTRAINT partsupp_ps_partkey_fkey FOREIGN KEY (ps_partkey) REFERENCES part (p_partkey) ON DELETE CASCADE;
ALTER TABLE partsupp ADD CONSTRAINT partsupp_ps_suppkey_fkey FOREIGN KEY (ps_suppkey) REFERENCES supplier (s_suppkey) ON DELETE CASCADE;
ALTER TABLE customer ADD CONSTRAINT customer_c_nationkey_fkey FOREIGN KEY (c_nationkey) REFERENCES nation (n_nationkey) ON DELETE CASCADE;
ALTER TABLE orders ADD CONSTRAINT orders_o_custkey_fkey FOREIGN KEY (o_custkey) REFERENCES customer (c_custkey) ON DELETE CASCADE;
ALTER TABLE lineitem ADD CONSTRAINT lineitem_l_orderkey_fkey FOREIGN KEY (l_orderkey) REFERENCES orders (o_orderkey) ON DELETE CASCADE;
ALTER TABLE lineitem ADD CONSTRAINT lineitem_l_partkey_l_suppkey_fkey FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey) ON DELETE CASCADE;

CREATE UNIQUE INDEX r_rk ON region (r_regionkey ASC);
CREATE UNIQUE INDEX n_nk ON nation (n_nationkey ASC);
CREATE INDEX n_rk ON nation (n_regionkey ASC);
CREATE UNIQUE INDEX p_pk ON part (p_partkey ASC);
CREATE UNIQUE INDEX s_sk ON supplier (s_suppkey ASC);
CREATE INDEX s_nk ON supplier (s_nationkey ASC);
CREATE INDEX ps_pk ON partsupp (ps_partkey ASC);
CREATE INDEX ps_sk ON partsupp (ps_suppkey ASC);
CREATE UNIQUE INDEX ps_pk_sk ON partsupp (ps_partkey ASC, ps_suppkey ASC);
CREATE UNIQUE INDEX ps_sk_pk ON partsupp (ps_suppkey ASC, ps_partkey ASC);
CREATE UNIQUE INDEX c_ck ON customer (c_custkey ASC);
CREATE INDEX c_nk ON customer (c_nationkey ASC);
CREATE UNIQUE INDEX o_ok ON orders (o_orderkey ASC);
CREATE INDEX o_ck ON orders (o_custkey ASC);
CREATE INDEX o_od ON orders (o_orderdate ASC);
CREATE INDEX l_ok ON lineitem (l_orderkey ASC);
CREATE INDEX l_pk ON lineitem (l_partkey ASC);
CREATE INDEX l_sk ON lineitem (l_suppkey ASC);
CREATE INDEX l_sd ON lineitem (l_shipdate ASC);
CREATE INDEX l_cd ON lineitem (l_commitdate ASC);
CREATE INDEX l_rd ON lineitem (l_receiptdate ASC);
CREATE INDEX l_pk_sk ON lineitem (l_partkey ASC, l_suppkey ASC);
CREATE INDEX l_sk_pk ON lineitem (l_suppkey ASC, l_partkey ASC);