SELECT OL_I_ID, OL_SUPPLY_W_ID, OL_QUANTITY,
OL_AMOUNT, OL_DELIVERY_D
FROM ORDER_LINE
WHERE OL_O_ID = 1
AND OL_D_ID = 1
AND OL_W_ID = 1;
