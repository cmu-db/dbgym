"""
To run, be in dbgym/demo/ and run `python query_html_gen.py`.
"""

def query_to_html(query: str) -> str:
    assert (
        "='" not in query
    ), "Some queries in the repo are incorrectly formatted this way"

    # Keywords
    keywords = [
        "SELECT",
        "FROM",
        "WHERE",
        "AND",
        "OR",
        "AS",
        "LIKE",
        "!=",
        " = ",
        " > ",
        "NULL",
        " IN ",
    ]
    for keyword in keywords:
        query = query.replace(keyword, f"<span class='query-red'>{keyword}</span>")

    # Functions
    functions = ["MIN", "MAX"]
    for function in functions:
        query = query.replace(
            function, f"<span class='query-lightblue'>{function}</span>"
        )

    # Tables
    tables = [
        "ct",
        "it",
        "lt",
        "t",
        "mc",
        "mi_idx",
        "cn",
        "n",
        "mk",
        "k",
        "ml",
        "ci",
        "mi",
    ]
    for table in tables:
        query = query.replace(
            f"{table}.", f"<span class='query-lightblue'>{table}</span>."
        )

    # Columns
    columns = [
        "note",
        "title",
        "production_year",
        "id",
        "movie_id",
        "company_type_id",
        "info_type_id",
        "kind",
        "info",
        "keyword_id",
        "link_type_id",
        "company_id",
        "country_code",
        "name",
        "keyword",
        "link",
        "production_year",
        "person_id",
    ]
    for column in columns:
        query = query.replace(
            f".{column}", f".<span class='query-lightblue'>{column}</span>"
        )

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
        "Bulgaria",
        "%sequel%",
        "[de]",
        "character-name-in-title",
        "rating",
        "5.0",
    ]
    for string in strings:
        query = query.replace(
            f"'{string}'", f"<span class='query-darkblue'>'{string}'</span>"
        )

    # Numbers
    numbers = ["1950", "2000", "2010", "2005"]
    for number in numbers:
        query = query.replace(
            f"{number}", f"<span class='query-lightblue'>{number}</span>"
        )

    return query


if __name__ == "__main__":
    query1a = """
SELECT MIN(mc.note) AS production_note,<br>
    &nbsp;MIN(t.title) AS movie_title,<br>
    &nbsp;MIN(t.production_year) AS movie_year<br>
FROM company_type AS ct,<br>
    &nbsp;info_type AS it,<br>
    &nbsp;movie_companies AS mc,<br>
    &nbsp;movie_info_idx AS mi_idx,<br>
    &nbsp;title AS t<br>
WHERE ct.kind = 'production companies'<br>
    &nbsp;AND it.info = 'top 250 rank'<br>
    &nbsp;AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'<br>
    &nbsp;AND (mc.note LIKE '%(co-production)%'<br>
        &nbsp;&nbsp;OR mc.note LIKE '%(presents)%')<br>
    &nbsp;AND ct.id = mc.company_type_id<br>
    &nbsp;AND t.id = mc.movie_id<br>
    &nbsp;AND t.id = mi_idx.movie_id<br>
    &nbsp;AND mc.movie_id = mi_idx.movie_id<br>
    &nbsp;AND it.id = mi_idx.info_type_id;"""

    query2a = """
SELECT MIN(t.title) AS movie_title<br>
FROM company_name AS cn,<br>
    &nbsp;keyword AS k,<br>
    &nbsp;movie_companies AS mc,<br>
    &nbsp;movie_keyword AS mk,<br>
    &nbsp;title AS t<br>
WHERE cn.country_code = '[de]'<br>
    &nbsp;AND k.keyword = 'character-name-in-title'<br>
    &nbsp;AND cn.id = mc.company_id<br>
    &nbsp;AND mc.movie_id = t.id<br>
    &nbsp;AND t.id = mk.movie_id<br>
    &nbsp;AND mk.keyword_id = k.id<br>
    &nbsp;AND mc.movie_id = mk.movie_id;"""
    
    query4a = """
SELECT MIN(mi_idx.info) AS rating,<br>
    &nbsp;MIN(t.title) AS movie_title<br>
FROM info_type AS it,<br>
    &nbsp;keyword AS k,<br>
    &nbsp;movie_info_idx AS mi_idx,<br>
    &nbsp;movie_keyword AS mk,<br>
    &nbsp;title AS t<br>
WHERE it.info = 'rating'<br>
    &nbsp;AND k.keyword LIKE '%sequel%'<br>
    &nbsp;AND mi_idx.info > '5.0'<br>
    &nbsp;AND t.production_year > 2005<br>
    &nbsp;AND t.id = mi_idx.movie_id<br>
    &nbsp;AND t.id = mk.movie_id<br>
    &nbsp;AND mk.movie_id = mi_idx.movie_id<br>
    &nbsp;AND k.id = mk.keyword_id<br>
    &nbsp;AND it.id = mi_idx.info_type_id;"""

    queries = [
        ("1a", query1a),
        ("2a", query2a),
        ("4a", query4a),
    ]

    for query_name, query in queries:
        html_query = query_to_html(query)
        with open(f"components/genned_query{query_name}.html", "w") as f:
            f.write(html_query)
