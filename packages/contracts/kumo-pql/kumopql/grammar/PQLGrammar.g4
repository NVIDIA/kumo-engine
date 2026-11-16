// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

grammar PQLGrammar;

prog:
	PREDICT target problem_spec
	FOR_EACH entity whatif EOF;

// Section: all non-terminal symbols.

problem_spec:
	forecast
	| problem_type top_k
	| top_k  // top_k alone is not valid - validation happens in problem_type_validator
	| problem_type
	|;

problem_type: (RANK | CLASSIFY);

top_k: (TOP INT);

forecast: (FORECAST INT TIMEFRAMES);

target: condition | aggregation | column;

entity: column | filtered_column;

whatif: (ASSUMING condition)?;

// antlr takes care of precedence from the order of the rules
condition:
	aggregation REL_OP constant
	| column REL_OP constant
	| NOT condition
	| condition AND condition
	| condition OR condition
	| '(' condition ')';

aggregation:
	AGGR '(' (column | filtered_column) ',' (INT | NEG_INFTY) ',' INT ')'
	| AGGR '(' (column | filtered_column) ',' (INT | NEG_INFTY) ',' INT ',' TIME_UNIT ')'
	| AGGR '(' (column | filtered_column) ')';

filtered_column: column WHERE condition;

// FQN stands for fully-qualified name
column: FQN_COLUMN | WILDCARD_COLUMN;

constant:
	BOOL
	| INT
	| STR
	| DECIMAL
	| datetime
	| NULL
	| array;

// Arrays of length 1 are disallowed to avoid ambiguity with a condition in parentheses. Arrays are
// only used in combination with "IS IN" which can be replaced with an '=' operator for a length-1
// array.
array:
	'(' (constant ',')+ constant ')'
	;

// Native boolean type
BOOL: 'true' | 'false' | 'True' | 'False' | 'TRUE' | 'FALSE';

// Section: all terminal symbols

FQN_COLUMN: ID '.' ID;
WILDCARD_COLUMN: ID '.' '*';

PREDICT: 'predict' | 'PREDICT';

CLASSIFY: 'classify' | 'CLASSIFY';

RANK: 'rank' | 'RANK';

TOP: 'top' | 'TOP';

FORECAST: 'forecast' | 'FORECAST';

TIMEFRAMES: 'timeframes' | 'TIMEFRAMES';

FOR_EACH: 'for each' | 'FOR EACH';

WHERE: 'where' | 'WHERE';

ASSUMING: 'assuming' | 'ASSUMING';

AND: 'and' | 'AND';

OR: 'or' | 'OR';

NOT: 'not' | 'NOT';

AGGR:
	'SUM'
	| 'AVG'
	| 'MIN'
	| 'MAX'
	| 'COUNT_DISTINCT'
	| 'FIRST'
	| 'LAST'
	| 'LIST_DISTINCT'
	| 'COUNT'
	| 'sum'
	| 'avg'
	| 'min'
	| 'max'
	| 'count_distinct'
	| 'first'
	| 'last'
	| 'list_distinct'
	| 'count';

REL_OP:
	'!='
	| '<='
	| '>='
	| '<'
	| '>'
	| '='
	| 'IS'
	| 'IS NOT'
	| 'IN'
	| 'IS IN'
	| 'LIKE'
	| 'NOT LIKE'
	| 'CONTAINS'
	| 'NOT CONTAINS'
	| 'STARTS WITH'
	| 'ENDS WITH'
	| 'is'
	| 'is not'
	| 'in'
	| 'is in'
	| 'like'
	| 'not like'
	| 'contains'
	| 'not contains'
	| 'starts with'
	| 'ends with';

NULL: 'NULL' | 'null';
NEG_INFTY: '-INF' | '-INFINITY' | '-inf' | '-infinity';
TIME_UNIT:
	'days'
	| 'months'
	| 'hours'
	| 'minutes'
	| 'DAYS'
	| 'MONTHS'
	| 'MINUTES'
	| 'HOURS';

// Section: all regular expressions that correspond to non-keyword symbols (identifiers, constants)

// We allow numbers in table/column names but demand the first character to be a letter or an
// underscore to avoid ambiguity with numbers. This follows SQL name guidelines.

ID: [_a-zA-Z][_a-zA-Z0-9]*;

INT: '-'? [0-9]+;

DECIMAL:
	'-'? INT '.' [0-9]+ EXP? // 1.35, 1.35E-9, 0.3, -4.5
	| '-'? INT EXP; // 1e10 -3e4
fragment EXP: [Ee] [+\-]? INT;

// This regex matches string descriptions in double or single quotation marks (first and second
// line, respectively). It correctly handles escaped quotation marks and special characters, e.g.
// newlines.
STR:
	'"' ('\\' ["\\/bfnrtu] | ~["])* '"'
	| '\'' ('\\' ['\\/bfnrtu] | ~['])* '\'';

// This code only approximately matches the dates and is not meant to validate them. We do that with
// Pandas later - validating dates with regex is possible but highly unpleasant.
datetime: DATE TIME?;

DATE:
	DATEINT '/' DATEINT '/' DATEINT
	| DATEINT '-' DATEINT '-' DATEINT
	| DATEINT '.' DATEINT '.' DATEINT;
TIME: DATEINT ':' DATEINT ':' DATEINT;
fragment DATEINT: [0-9]+;

// ignore any unmatched whitespace and comments
WS : ([ \t\n\r\u000b\u000c\u001c\u001d\u001e\u001f\u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000]+ | COMMENT) -> skip ;

fragment COMMENT: '#' ~('\r' | '\n')*;
