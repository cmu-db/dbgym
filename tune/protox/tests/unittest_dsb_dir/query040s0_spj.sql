select
   min(w_state)
  ,min(i_item_id)
  ,min(cs_item_sk)
  ,min(cs_order_number)
  ,min(cr_item_sk)
  ,min(cr_order_number)
 from
   catalog_sales left outer join catalog_returns on
       (cs_order_number = cr_order_number
        and cs_item_sk = cr_item_sk)
  ,warehouse
  ,item
  ,date_dim
 where
 i_item_sk          = cs_item_sk
 and cs_warehouse_sk    = w_warehouse_sk
 and cs_sold_date_sk    = d_date_sk
 and d_date between (cast ('1998-04-26' as date) - interval '30' day)
                and (cast ('1998-04-26' as date) + interval '30' day)
 and i_category  = 'Home'
 and i_manager_id between 28 and 67
 and cs_wholesale_cost between 69 and 88
 and cr_reason_sk = 11;
