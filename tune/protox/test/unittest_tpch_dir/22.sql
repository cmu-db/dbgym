
select
	cntrycode,
	count(*) as numcust,
	sum(c_acctbal) as totacctbal
from
	(
		select
			substring(c_phone from 1 for 2) as cntrycode,
			c_acctbal
		from
			customer c1
		where
			substring(c_phone from 1 for 2) in
				('10', '14', '11', '30', '29', '21', '12')
			and c_acctbal > (
				select
					avg(c_acctbal)
				from
					customer c2
				where
					c_acctbal > 0.00
					and substring(c_phone from 1 for 2) in
						('10', '14', '11', '30', '29', '21', '12')
			)
			and not exists (
				select
					*
				from
					orders
				where
					o_custkey = c_custkey
			)
	) as custsale
group by
	cntrycode
order by
	cntrycode;
