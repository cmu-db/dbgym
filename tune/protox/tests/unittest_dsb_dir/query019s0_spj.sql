select min(i_brand_id), min(i_manufact_id),
 	min(ss_ext_sales_price)
 from date_dim, store_sales, item,customer,customer_address,store
 where d_date_sk = ss_sold_date_sk
   and ss_item_sk = i_item_sk
   and ss_customer_sk = c_customer_sk
   and c_current_addr_sk = ca_address_sk
   and ss_store_sk = s_store_sk
   AND i_category  = 'Home'
   and d_year=2002
   and d_moy = 8
   and substring(ca_zip,1,5) <> substring(s_zip,1,5)
   and ca_state  = 'GA'
   and c_birth_month = 4
   and ss_wholesale_cost BETWEEN 80 AND 100;