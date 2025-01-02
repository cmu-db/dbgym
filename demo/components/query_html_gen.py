def query_to_html(query: str) -> str:
    # Keywords
    keywords = ["SELECT", "FROM", "WHERE", "AND", "OR", "AS", "LIKE"]
    for keyword in keywords:
        query = query.replace(keyword, f"<span class='query-red'>{keyword}</span>")

    # Functions
    functions = ["MIN", "MAX"]
    for function in functions:
        query = query.replace(function, f"<span class='query-lightblue'>{function}</span>")

    # Tables
    tables = ["ct", "it", "mc", "mi_idx", "t"]
    for table in tables:
        query = query.replace(f"{table}.", f"<span class='query-lightblue'>{table}</span>.")

    # Columns
    columns = ["note", "title", "production_year", "id", "movie_id", "company_type_id", "info_type_id", "kind", "info"]
    for column in columns:
        query = query.replace(f".{column}", f".<span class='query-lightblue'>{column}</span>")

    # Strings
    query = query.replace(" '", " <span class='query-darkblue'>'")
    query = query.replace("' ", "'</span> ")
    query = query.replace("')", "'</span>)")

    return query


if __name__ == "__main__":
    query = """SELECT MIN(mc.note) AS production_note,
       MIN(t.title) AS movie_title,
       MIN(t.production_year) AS movie_year<br>
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info_idx AS mi_idx,
     title AS t<br>
WHERE ct.kind = 'production companies'
  AND it.info = 'top 250 rank'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND (mc.note LIKE '%(co-production)%'
       OR mc.note LIKE '%(presents)%')
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id;"""
    html_query = query_to_html(query)
    print(html_query)