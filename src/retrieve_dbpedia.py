# -*- coding: utf-8 -*-
"""DBpediaからSPARQLで一覧を取得しファイルに出力する"""


import re
import sys
import argparse

from SPARQLWrapper import SPARQLWrapper

BRACKET_PATTERN = re.compile("^(.+) \(.+\)")


def create_parser():
    parser = argparse.ArgumentParser()

    # SPARQLファイル
    parser.add_argument(
        'query',
        type=argparse.FileType('r'),
        help="SPARQLファイル"
    )

    # 出力ファイルパス(デフォルトはstdout)
    parser.add_argument(
        'outfile',
        nargs='?',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="結果の出力先(デフォルトはstdout)"
    )

    return parser


def remove_last_bracket(org):
    """末尾の丸括弧を外す

    例: 「市役所前駅 (和歌山県)」→「市役所前駅」
    """

    matcher = BRACKET_PATTERN.match(org)
    if matcher:
        return matcher.group(1)

    return org


def main(args):
    # DBpediaから取得
    sparql = SPARQLWrapper(endpoint='http://ja.dbpedia.org/sparql', returnFormat='json')
    sparql.setQuery(args.query.read())
    results = sparql.query().convert()

    # 読み込み
    stations = [remove_last_bracket(record["name"]["value"])
                for record in results["results"]["bindings"]]
    # 書き出し
    with args.outfile as f:
        for s in stations:
            f.write(s)
            f.write("\n")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
