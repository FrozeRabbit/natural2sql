import sqlite3
import os

DB_PATH = "test/database/bioinfo.db"

# DDL
DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS Organism (
        organism_id   INTEGER PRIMARY KEY,
        genus         TEXT    NOT NULL,
        species       TEXT    NOT NULL,
        common_name   TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Gene (
        gene_id       INTEGER PRIMARY KEY,
        organism_id   INTEGER NOT NULL,
        gene_symbol   TEXT    NOT NULL,
        description   TEXT,
        FOREIGN KEY (organism_id) REFERENCES Organism(organism_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Sequence (
        sequence_id    INTEGER PRIMARY KEY,
        gene_id        INTEGER NOT NULL,
        seq_type       TEXT    CHECK(seq_type IN ('DNA','RNA','Protein')) NOT NULL,
        sequence_text  TEXT    NOT NULL,
        FOREIGN KEY (gene_id) REFERENCES Gene(gene_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Variant (
        variant_id       INTEGER PRIMARY KEY,
        gene_id          INTEGER NOT NULL,
        variant_name     TEXT    NOT NULL,
        position         INTEGER NOT NULL,
        reference_base   TEXT    NOT NULL,
        alternate_base   TEXT    NOT NULL,
        FOREIGN KEY (gene_id) REFERENCES Gene(gene_id)
    );
    """
]

# 示例数据
EXAMPLE_DATA = {
    "Organism": [
        # (organism_id, genus, species, common_name)
        (1, "Homo", "sapiens", "Human"),
        (2, "Mus", "musculus", "Mouse")
    ],
    "Gene": [
        # (gene_id, organism_id, gene_symbol, description)
        (1, 1, "BRCA1", "Breast cancer type 1 susceptibility protein"),
        (2, 1, "TP53",  "Tumor protein p53"),
        (3, 2, "Trp53", "Mouse tumor protein p53")
    ],
    "Sequence": [
        # (sequence_id, gene_id, seq_type, sequence_text)
        (1, 1, "DNA",     "ATGGATTTTCCTGCCAGTGTGACTTGCGCTCTGATTGGCT"),
        (2, 1, "Protein", "MDLSALSVKAKLLSP"),  
        (3, 2, "DNA",     "ATGTTCCAGATTACAGCCATGCACAGGTGCA"),
        (4, 2, "Protein", "MPVSRRKAIRDYNYTRL"),
        (5, 3, "DNA",     "ATGCCCTCAGAGTCTCCTGGTG"),
        (6, 3, "Protein", "MEEPQSDPSVEPPLSQ"),
    ],
    "Variant": [
        # (variant_id, gene_id, variant_name, position, reference_base, alternate_base)
        (1, 1, "rs80357065",  43071077, "C", "T"),  
        (2, 2, "rs1042522",   7579472,  "G", "A"),  
        (3, 3, "rs57873645",  7579472,  "G", "T")   
    ]
}

def create_database(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for ddl in DDL_STATEMENTS:
        cursor.execute(ddl)

    conn.commit()
    return conn

def populate_example_data(conn: sqlite3.Connection):

    cursor = conn.cursor()

    cursor.executemany(
        "INSERT OR IGNORE INTO Organism (organism_id, genus, species, common_name) VALUES (?, ?, ?, ?);",
        EXAMPLE_DATA["Organism"]
    )

    cursor.executemany(
        "INSERT OR IGNORE INTO Gene (gene_id, organism_id, gene_symbol, description) VALUES (?, ?, ?, ?);",
        EXAMPLE_DATA["Gene"]
    )

    cursor.executemany(
        "INSERT OR IGNORE INTO Sequence (sequence_id, gene_id, seq_type, sequence_text) VALUES (?, ?, ?, ?);",
        EXAMPLE_DATA["Sequence"]
    )

    cursor.executemany(
        "INSERT OR IGNORE INTO Variant (variant_id, gene_id, variant_name, position, reference_base, alternate_base) VALUES (?, ?, ?, ?, ?, ?);",
        EXAMPLE_DATA["Variant"]
    )

    conn.commit()

def main():
    conn = create_database(DB_PATH)
    populate_example_data(conn)
    conn.close()

if __name__ == "__main__":
    main()
