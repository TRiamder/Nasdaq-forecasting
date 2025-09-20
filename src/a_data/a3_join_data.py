import sqlite3
import pandas as pd

def join_data():
    # SQL-query to join the data tables
    query = """ 
                WITH full_dates AS (
                SELECT "date" FROM nasdaq_prices
                UNION 
                SELECT DATE("DATE") FROM treasury_yield_10y
                UNION 
                SELECT DATE("DATE") FROM fed_funds_rate
                UNION 
                SELECT DATE("DATE") FROM cpi 
                UNION 
                SELECT DATE("DATE") FROM unemployment_rate
                )
                SELECT 
                    f.date, 
                    n.open, n.high, n.low, n.close, n.volume, 
                    y.DGS10, 
                    r.DFF, 
                    p.CPIAUCSL, 
                    u.UNRATE 
                FROM full_dates AS f
                LEFT JOIN nasdaq_prices AS n ON f.date = n.date
                LEFT JOIN treasury_yield_10y AS y ON f.date = DATE(y.DATE) 
                LEFT JOIN fed_funds_rate AS r ON f.date = DATE(r.DATE) 
                LEFT JOIN cpi AS p ON f.date = DATE(p.DATE) 
                LEFT JOIN unemployment_rate AS u ON f.date = DATE(u.DATE) 
                ORDER BY f.date
    """

    # save the joined tables in the database
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        joined = pd.read_sql_query(query, conn)
        joined.to_sql('nasdaq_macros_joined', conn, if_exists='replace', index=False)

    #print(joined.head(10))
    #print(joined.info())

if __name__ == '__main__':
    join_data()
