## Updating Autocomplete Grammar
The original implementation of the predictive query code editor ([CodeMirror](https://uiwjs.github.io/react-codemirror/)) relies on defining a language object that utilizes a stream parser for our custom PQ language. That stream parser had limiting capabilities (e.g. cannot support sophisticated autocompletion) and, additionally, cannot take advantage of our language grammar. Within CodeMirror’s ecosystem of libraries, an alternative option is to define a language with [LRParser](https://lezer.codemirror.net/docs/ref/#lr.LRParser). However, this parser is not compatible with the existing `PQLGrammar.g4` (ANTLR compatible grammar) file. These two points motivated a solution to bridge the gap between our existing grammar and CodeMirror.

### Solution Outline
1. Use [ANTLR (for TypeScript; antlr4ts)](https://github.com/tunnelvisionlabs/antlr4ts) to generate all of the lexer and parser files based on `PQLGrammar.g4`.
2. Define a language server stored in `PQLServer.ts`.
3. Define a parser adapter stored in `PQLParserAdapter.ts`.
4. Define a PQL1 `Language` object stored in `PQL1.ts`. `CodeEditor` currently uses this `PQL1` language.
5. Implement keyword and function (snippet) autocompletion.

### Detailed Explanation
1. Use `antlr4ts` to generate all the lexer and parser files based on the `PQLGrammar.g4` file.
* Run `npm --prefix ui run compile-grammar-to-ts` from the repository root to generate files.
2. Define a language server.
- Define the token types that the language server supports.
- Tokenize the lexer's tokens: Create a mapping between the token types and the lexer’s types. Originally, the lexer’s tokens map to numbers, but by defining the token types with string, we can map them to a string instead.
- Define a simple language server class — PQLServer — with two methods. `getTokenStream` creates the token stream used to build a tree in the parser adapter. `getTokenTypeForIndex` is a helper method to map between the lexer token types and their string types.
3. Define a parser adapter, as an extension to Lezer/CodeMirror’s Parser class. The purpose of the parser adapter is to bridge ANTLR and Lezer. The Lezer parser defines two functions — `startParse` and `createParse` — to generate a syntax tree. The syntax tree is a collection of nodes (`NodeSet`) with type `NodeType`. More info is available in the [official docs](https://lezer.codemirror.net/docs/ref/#common.Trees). Parser adapter actions:
- Define the `tokenToNodeType` mapping in `PQLConstants.ts`, imported by the adapter, to map each token type to NodeType with a unique index id. Because `parserAdapterNodeSet` is built from `Object.values(tokenToNodeType)`, keep the mapping in ID order: each `NodeType.id` must match its index in the `NodeSet` array. The 0-th node is “document”, marked as a top node, is _not a token type previously defined in PQL_ and is only used as an entry point in the tree by Lezer.
- Create a set (`NodeSet`) of all `NodeTypes` in `PQLConstants.ts`. While creating the `NodeSet`, I attached an extension where I assigned the highlight tags for each `NodeType`.
- Create a buffer of node information to build the tree. Lezer’s Tree class accepts either a BufferCursor or a buffer of integers to read the node data from. The buffer of integers is read in sets of four.
From the [official docs](https://lezer.codemirror.net/docs/ref/#common.Tree^build^data.buffer): 

> When this [buffer] is an array, it should contain four values for every node in the tree.
> - The first holds the node's type, as a node ID pointing into the given NodeSet.
> - The second holds the node's start offset. [Relative to the start of the document. The top node has a start of 0 and end of 11.]
> - The third the end offset.
> - The fourth the amount of space taken up in the array by this node and its children. Since there's four values per node, this is the total number of nodes inside this node (children and transitive children) plus one for the node itself, times four.

For simplicity, I defined the fourth value as a default size of 4. Additionally, ANTLR’s stop index is at the last character of the word and Lezer’s stop index refers to the first character after the last character of the word. To account for the difference, I have defined `endOffset = stopIndex + 1`.
- Define `buildTree` method — get the token stream from the language server -> create a buffer -> build Tree with buffer, NodeSet, and top node.
- Implement  `startParse` and `createParse`. `createParse` is used for initial tree building and `startParse` is used during code editing.
5. Define a language with language data, parser adapter, extra extensions, and language name. The language data is `Facet.define()`. More information about the `Facet` class if available in the official [docs](https://codemirror.net/docs/ref/#state.Facet), but essentially the language data is defined and passed through the top node's `props` argument in the `tokenToNodeType` mapping. Additionally, the  highlighting style for syntax highlighting is defined and passed as an extension.
5. Autocompletion:
* Steps: 1. Parse the user input to only get the latest "word" or partial "word"; 2. Access the syntax tree at the cursor position; 3. Analyze the tree to determine the current context; 4. Limit what completions can be based on context.
* Completion Source: The list of keyword completions is available in the `snippetsAndKeywords` file; whereas, a list of table and column names is generated when the user selects the graph.
* Summary: Creating a parser adapter (and all other supporting files) enables us to parse the editor state and create a syntax tree based on the cursor position (step 2 of autocompletion).

### Grammar dependency
As outlined in step one from the previous section, the code files (lexer, parser, listener, etc. stored in `ui/src/utils/PQLParsing/grammar`) are derived/generated from `PQLGrammar.g4`. Most likely, any significant changes to the `g4` file will also affect the code files. As a general rule of thumb, if the code files in the backend need to be regenerated, so will the files in the ui directory. Also, if any keywords are removed/added to PQL, the array of autocompletion keywords in `ui/src/utils/PQLParsing/snippetsAndKeywords.ts` needs to be updated accordingly.

### Resources
[Building autocomplete with ANTLR and CodeMirror](https://www.sumologic.com/blog/building-autocomplete-antlr-codemirror/)
[Connecting ANTLR to CodeMirror 6: Building a Language Server](https://thetrevorharmon.com/blog/connecting-antlr-to-codemirror-6-building-a-language-server/)
[Connecting ANTLR to CodeMirror 6: Connecting a Language Server](https://thetrevorharmon.com/blog/connecting-antlr-to-code-mirror-6-connecting-a-language-server/)
[Lezer Docs](https://lezer.codemirror.net/docs/)
[Lezer Parser Source Code](https://github.com/lezer-parser/common/blob/main/src/parse.ts)
[CodeMirror Docs](https://codemirror.net/docs/) (especially [Writing a Language Package](https://codemirror.net/examples/lang-package/), [@codemirror/language](https://codemirror.net/docs/ref/#language), [@codemirror/state](https://codemirror.net/docs/ref/#state))
[@codemirror/lang-javascript source code](https://github.com/codemirror/lang-javascript)
