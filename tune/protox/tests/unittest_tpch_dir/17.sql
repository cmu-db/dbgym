
select
	sum(l_extendedprice) / 7.0 as avg_yearly
from
	lineitem l1,
	part
where
	p_partkey = l_partkey
	and p_brand = 'Brand#22'
	and p_container = 'SM BAG'
	and l_quantity < (
		select
			0.2 * avg(l_quantity)
		from
			lineitem l2
		where
			l_partkey = p_partkey
	);
