SELECT C_FIRST, C_MIDDLE, C_LAST, C_STREET_1,
C_STREET_2, C_CITY, C_STATE, C_ZIp, C_PHONE,
C_CREDIT, C_CREDIT_LIM, C_DISCOUNT, C_BALANCE,
C_YTD_PAYMENT, C_PAYMENT_CNT, C_SINCE
FROM CUSTOMER
WHERE C_W_ID = 1
AND C_D_ID = 1
AND C_ID = 40;
