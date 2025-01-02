def query_to_html(query: str) -> str:
    assert "='" not in query, "Some queries in the repo are incorrectly formatted this way"

    # Keywords
    keywords = ["SELECT", "FROM", "WHERE", "AND", "OR", "AS", "LIKE", "!=", " = ", " > ", "NULL"]
    for keyword in keywords:
        query = query.replace(keyword, f"<span class='query-red'>{keyword}</span>")

    # Functions
    functions = ["MIN", "MAX"]
    for function in functions:
        query = query.replace(function, f"<span class='query-lightblue'>{function}</span>")

    # Tables
    tables = ["ct", "it", "lt", "t", "mc", "mi_idx", "cn", "n", "mk", "k", "ml", "ci"]
    for table in tables:
        query = query.replace(f"{table}.", f"<span class='query-lightblue'>{table}</span>.")

    # Columns
    columns = [
        "note", "title", "production_year", "id", "movie_id", "company_type_id", "info_type_id", "kind", "info",
        "keyword_id", "link_type_id", "company_id", "country_code", "name", "keyword", "link",
        "production_year", "person_id"
    ]
    for column in columns:
        query = query.replace(f".{column}", f".<span class='query-lightblue'>{column}</span>")

    # Strings
    strings = [
        "production companies",
        "top 250 rank",
        "%(as Metro-Goldwyn-Mayer Pictures)%",
        "%(co-production)%",
        "%(presents)%",
        "[pl]",
        "%Film%",
        "%Warner%",
        "sequel",
        "%follow%",
        "marvel-cinematic-universe",
        "%Downey%Robert%",
    ]
    for string in strings:
        query = query.replace(f"'{string}'", f"<span class='query-darkblue'>'{string}'</span>")

    # Numbers
    numbers = [
        "1950",
        "2000",
        "2010"
    ]
    for number in numbers:
        query = query.replace(f"{number}", f"<span class='query-lightblue'>{number}</span>")

    return query


if __name__ == "__main__":
    query1a = """SELECT MIN(mc.note) AS production_note,
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
    

    query6a = """SELECT MIN(k.keyword) AS movie_keyword,
       MIN(n.name) AS actor_name,
       MIN(t.title) AS marvel_movie<br>
FROM cast_info AS ci,
     keyword AS k,
     movie_keyword AS mk,
     name AS n,
     title AS t<br>
WHERE k.keyword = 'marvel-cinematic-universe'
  AND n.name LIKE '%Downey%Robert%'
  AND t.production_year > 2010
  AND k.id = mk.keyword_id
  AND t.id = mk.movie_id
  AND t.id = ci.movie_id
  AND ci.movie_id = mk.movie_id
  AND n.id = ci.person_id;"""

    query11a = """SELECT MIN(cn.name) AS from_company,
       MIN(lt.link) AS movie_link_type,
       MIN(t.title) AS non_polish_sequel_movie<br>
FROM company_name AS cn,
     company_type AS ct,
     keyword AS k,
     link_type AS lt,
     movie_companies AS mc,
     movie_keyword AS mk,
     movie_link AS ml,
     title AS t<br>
WHERE cn.country_code != '[pl]'
  AND (cn.name LIKE '%Film%'
       OR cn.name LIKE '%Warner%')
  AND ct.kind = 'production companies'
  AND k.keyword = 'sequel'
  AND lt.link LIKE '%follow%'
  AND mc.note IS NULL
  AND t.production_year BETWEEN 1950 AND 2000
  AND lt.id = ml.link_type_id
  AND ml.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND t.id = mc.movie_id
  AND mc.company_type_id = ct.id
  AND mc.company_id = cn.id
  AND ml.movie_id = mk.movie_id
  AND ml.movie_id = mc.movie_id
  AND mk.movie_id = mc.movie_id;"""
    
    queries = [("1a", query1a), ("6a", query6a), ("11a", query11a)]

    for query_name, query in queries:
        html_query = query_to_html(query)
        with open(f"components/genned_query{query_name}.html", "w") as f:
            f.write(html_query)
