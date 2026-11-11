# EDA – Contest_Data

## 1. Documents per country × knowledge base × law

| country   |   ('cases', 'Divorce') |   ('cases', 'Inheritance') |   ('divorce_code', 'Divorce') |   ('inheritance_code', 'Inheritance') |
|:----------|-----------------------:|---------------------------:|------------------------------:|--------------------------------------:|
| Estonia   |                     91 |                         31 |                            50 |                                    49 |
| Italy     |                    129 |                        341 |                            38 |                                   302 |
| Slovenia  |                    101 |                        100 |                            30 |                                   209 |

Total loaded: **1471** documents; skipped: **1**; after chunking: **1656** chunks.

- skipped `Article_815-819.json`: text shorter than 30 chars

## 2. Text length (characters, after cleaning)

|                      |   count |   min |   50% |   max |
|:---------------------|--------:|------:|------:|------:|
| ('case', 'Estonia')  |     122 |   546 |  1376 |  3597 |
| ('case', 'Italy')    |     470 |   306 |   942 |  2638 |
| ('case', 'Slovenia') |     201 |   467 |  1080 |  3255 |
| ('code', 'Estonia')  |      99 |   107 |   695 |  2914 |
| ('code', 'Italy')    |     340 |    59 |   370 |  1537 |
| ('code', 'Slovenia') |     239 |    71 |   427 |  2941 |

Documents longer than 1,500 characters (split into chunks): codes 18, cases 131.

## 3. Past cases: metadata coverage (% of cases with a value)

|                              |   Estonia |   Italy |   Slovenia |
|:-----------------------------|----------:|--------:|-----------:|
| duration                     |       100 |      67 |         93 |
| cost                         |        52 |      60 |         76 |
| subject_of_succession        |        14 |      53 |         48 |
| disputed_issues              |        17 |      48 |         21 |
| number_of_persons_involved   |        14 |      43 |         48 |
| succession_type              |         7 |      19 |         42 |
| relationship_between_parties |         6 |      16 |         23 |
| financial_support            |        24 |      16 |         17 |
| marital_regime               |        38 |      10 |         15 |
| presence_of_children         |        13 |       6 |          2 |
| nature_of_separation         |        22 |       4 |          5 |
| duration_of_marriage         |        14 |       2 |          3 |
| testamentary_clauses         |         2 |       2 |          1 |

Cases per law and proceeding type:

|                             |   extrajudicial |   judicial |
|:----------------------------|----------------:|-----------:|
| ('Estonia', 'Divorce')      |               0 |         91 |
| ('Estonia', 'Inheritance')  |               0 |         31 |
| ('Italy', 'Divorce')        |               3 |        126 |
| ('Italy', 'Inheritance')    |             124 |        217 |
| ('Slovenia', 'Divorce')     |               0 |        101 |
| ('Slovenia', 'Inheritance') |               0 |        100 |

## 4. Article number: text header vs `civil_codes_used`

In **41 / 678** code articles the number in the first line of the text differs from `civil_codes_used`: the metadata holds an article *referenced* in the text. The loader uses the header as `article` and keeps the metadata in `articles_cited`.

| country | kb | file | article (text) | civil_codes_used |
|---|---|---|---|---|
| Estonia | divorce_code | Article_Subchapter 2.json | 24 | 2 |
| Estonia | divorce_code | Article_§ 15 and § 16.json | 41 | 15 |
| Estonia | divorce_code | Article_§ 28.json | 30 | 28 |
| Estonia | divorce_code | Article_§ 36.json | 35 | 36 |
| Estonia | divorce_code | Article_§ 41.json | 43 | 41 |
| Estonia | divorce_code | Article_§§ 68 and 69.json | 70 | 68 |
| Estonia | inheritance_code | Article_§ 89.json | 93 | 89 |
| Italy | inheritance_code | Article_1427 et seq..json | 768-QUINQUIES | 1427 |
| Italy | inheritance_code | Article_279.json | 580 | 279 |
| Italy | inheritance_code | Article_561.json | 563 | 561 |
| Italy | inheritance_code | Article_592, 593, 595, 596, 597, 598.json | 599 | 592 |
| Italy | inheritance_code | Article_620, 621.json | 622 | 620 |
| Italy | inheritance_code | Article_720, 722.json | 727 | 720 |
| Italy | inheritance_code | Article_ART279.json | 594 | 279 |
| Italy | inheritance_code | Article_ART416.json | 692 | 416 |

## 5. Most cited articles in past cases

- **Italy**: Art. 720 (123), Art. 729 (51), Art. 727 (47), Art. 713 (46), Art. 718 (40), Art. 1116 (27), Art. 737 (26), Art. 1111 (24)
- **Estonia**: Art. 37 (3), Art. 142 (2), Art. 317 (2), Art. 444 (2), Art. 468 (2), KRS (1), TsÜS (1), AÕS (1)
- **Slovenia**: Art. 11 (74), Art. 59 (51), Art. 51 (38), Art. 163 (23), Art. 10 (21), Art. 279 (19), Art. 367 (17), Art. 8 (11)
