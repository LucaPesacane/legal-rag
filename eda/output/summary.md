# EDA - Contest_Data

## Documents per country and knowledge base
| country   |   ('cases', 'Divorce') |   ('cases', 'Inheritance') |   ('divorce_code', 'Divorce') |   ('inheritance_code', 'Inheritance') |
|:----------|-----------------------:|---------------------------:|------------------------------:|--------------------------------------:|
| Estonia   |                     91 |                         31 |                            50 |                                    49 |
| Italy     |                    129 |                        341 |                            38 |                                   302 |
| Slovenia  |                    101 |                        100 |                            30 |                                   209 |

Documents: 1471 - chunks after splitting: 1629

## Text length (characters)
|                      |   min |   50% |   max |
|:---------------------|------:|------:|------:|
| ('case', 'Estonia')  |   546 |  1376 |  3597 |
| ('case', 'Italy')    |   306 |   942 |  2638 |
| ('case', 'Slovenia') |   467 |  1080 |  3255 |
| ('code', 'Estonia')  |   107 |   695 |  2914 |
| ('code', 'Italy')    |    59 |   370 |  1537 |
| ('code', 'Slovenia') |    71 |   427 |  2941 |

Documents longer than 1500 characters (split): 149

## Past cases: % with metadata field
|                            |   Estonia |   Italy |   Slovenia |
|:---------------------------|----------:|--------:|-----------:|
| duration                   |       100 |      67 |         93 |
| cost                       |        52 |      60 |         76 |
| subject_of_succession      |        14 |      53 |         48 |
| disputed_issues            |        17 |      48 |         21 |
| succession_type            |         7 |      19 |         42 |
| number_of_persons_involved |        14 |      43 |         48 |
| marital_regime             |        38 |      10 |         15 |
| financial_support          |        24 |      16 |         17 |
| presence_of_children       |        13 |       6 |          2 |

## Article number: text vs civil_codes_used
In 42 of 679 code articles civil_codes_used is an article cited in the text, not the article itself: the loader reads the number from the text.
