SELECT S_QUANTITY, S_DATA, S_DIST_01, S_DIST_02,
S_DIST_03, S_DIST_04, S_DIST_05,
S_DIST_06, S_DIST_07, S_DIST_08,
S_DIST_09, S_DIST_10 FROM STOCK
WHERE S_I_ID = 1
AND S_W_ID = 1 FOR UPDATE;
