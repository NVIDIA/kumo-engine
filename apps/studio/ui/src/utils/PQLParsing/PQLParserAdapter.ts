// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Input, Parser, PartialParse, Tree, TreeFragment } from "@lezer/common";
import { Token } from "antlr4ts";

import { parserAdapterNodeSet, tokenToNodeType } from "./PQLConstants";
import { PQLServer } from "./PQLServer";

// Parser adapter to connect ANTLR grammar and Lezer parser.
export class ParserAdapter extends Parser {
    DEFAULT_NODE_GROUP_SIZE = 4;

    // Derive the node type from a token’s index; used to construct a tree.
    private languageServer = new PQLServer();
    private getNodeTypeIdForTokenIndex(index: number) {
        const tokenType = this.languageServer.getTokenTypeForIndex(index);
        return tokenToNodeType[tokenType].id;
    }

    /**
     * Buffer of integers that Lezer reads in groups of four.
     * Each group of four represents a single node.
     * The four integeres represent: 1. NodeType id, 2. NodeType start offset, 3. NodeType end offset, 4. The number of positions taken up by a node in the buffer. Smallest vlaue can be four.
     * The buffer (always?) has a top node that is the parent of the previous nodes.
     * Children nodes come before parent nodes.
     * The PQL language server returns the tokens in the order they appear in the document.
     */
    private createBufferFromTokens(tokens: Token[]) {
        const buffer: number[] = [];

        // Iterate over each token and push it into the buffer.
        tokens.forEach((token) => {
            const nodeTypeId = this.getNodeTypeIdForTokenIndex(token.type);
            const startOffset = token.startIndex;
            const endOffset = token.stopIndex + 1;
            buffer.push(
                nodeTypeId,
                startOffset,
                endOffset,
                this.DEFAULT_NODE_GROUP_SIZE
            );
        });

        const topNodeId = tokenToNodeType.document.id;
        const startOffset = tokens[0].startIndex;
        const endOffset = tokens[tokens.length - 1].stopIndex + 1;
        const topNodeSize =
            tokens.length * this.DEFAULT_NODE_GROUP_SIZE +
            this.DEFAULT_NODE_GROUP_SIZE;

        buffer.push(topNodeId, startOffset, endOffset, topNodeSize);
        return buffer;
    }

    buildTree(document: string) {
        const tokens = this.languageServer.getTokenStream(document);
        // Prevent an error from happening by building a minimal tree with top node.
        if (tokens.length < 1) {
            return Tree.build({
                buffer: [
                    tokenToNodeType.document.id,
                    0,
                    document.length,
                    this.DEFAULT_NODE_GROUP_SIZE
                ],
                nodeSet: parserAdapterNodeSet,
                topID: tokenToNodeType.document.id
            });
        }
        const buffer = this.createBufferFromTokens(tokens);

        return Tree.build({
            buffer: buffer,
            nodeSet: parserAdapterNodeSet,
            topID: tokenToNodeType.document.id
        });
    }

    createParse(
        input: Input,
        fragments: TreeFragment[],
        ranges: readonly { from: number; to: number }[]
    ): PartialParse {
        return this.startParse(input, fragments, ranges);
    }

    startParse(
        input: string | Input,
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        _fragments?: readonly TreeFragment[] | undefined,
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        _ranges?: readonly { from: number; to: number }[] | undefined
    ): PartialParse {
        const document =
            typeof input === "string" ? input : input.read(0, input.length);

        const tree = this.buildTree(document);
        return {
            stoppedAt: input.length,
            parsedPos: input.length,
            stopAt: () => {},
            advance: () => tree
        };
    }
}
