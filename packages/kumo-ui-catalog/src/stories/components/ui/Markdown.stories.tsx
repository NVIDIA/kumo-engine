// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Markdown } from "@/components/ui/markdown";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
    title: "UI/Markdown",
    component: Markdown,
    parameters: {
        layout: "padded"
    },
    tags: ["autodocs"],
    argTypes: {
        size: {
            control: "select",
            options: ["sm", "base", "lg"]
        },
        theme: {
            control: "select",
            options: ["default", "dark"]
        },
        enableSyntaxHighlighting: {
            control: "boolean",
            description:
                "Enable syntax highlighting for code blocks (default: false)"
        },
        enableCopyButton: {
            control: "boolean",
            description: "Enable copy button for code blocks (default: true)"
        },
        enableHeadingAnchors: {
            control: "boolean",
            description: "Enable anchor links for headings (default: false)"
        },
        enableLineNumbers: {
            control: "boolean",
            description: "Enable line numbers for code blocks (default: false)"
        },
        children: {
            control: "text"
        }
    }
} satisfies Meta<typeof Markdown>;

export default meta;
type Story = StoryObj<typeof meta>;

const basicMarkdown = `# Hello World

This is a simple paragraph with **bold text** and *italic text*.

Here's a [link to Google](https://google.com).`;

const comprehensiveMarkdown = `# Markdown Component Demo

This component supports a wide variety of Markdown features.

## Headings

You can use different heading levels:

### Third Level Heading
#### Fourth Level Heading

## Text Formatting

You can make text **bold**, *italic*, or ***both***.

## Lists

### Unordered Lists

- First item
- Second item
- Third item
  - Nested item 1
  - Nested item 2

### Ordered Lists

1. First step
2. Second step
3. Third step

## Code

Inline code: \`const greeting = "Hello World";\`

Code blocks:

\`\`\`javascript
function greet(name) {
  return \`Hello, \${name}!\`;
}

console.log(greet("World"));
\`\`\`

## Blockquotes

> This is a blockquote. It can span multiple lines and is useful for highlighting important information or quotes.

## Links

Visit [React MarkDown](https://github.com/remarkjs/react-markdown) for more information.

## Horizontal Rules

---

## Tables

| Feature | Supported | Notes |
|---------|-----------|-------|
| Headers | Yes | H1-H6 (all levels) |
| Lists | Yes | Ordered, Unordered & Task Lists |
| Code | Yes | Inline, Blocks & Syntax Highlighting |
| Tables | Yes | Full support with alignment |
| GFM | Yes | Strikethrough, autolinks, etc. |
| Images | Yes | With lazy loading |

---

That's all for now!`;

const codeExampleMarkdown = `# Code Examples

## JavaScript

\`\`\`javascript
const fetchData = async () => {
  const response = await fetch('https://api.example.com/data');
  const data = await response.json();
  return data;
};
\`\`\`

## Python

\`\`\`python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
\`\`\`

## Inline Code

You can use inline code like \`npm install react-markdown\` within text.`;

const tableExampleMarkdown = `# Table Example

Here's a comparison of different programming languages:

| Language | Type | Release Year | Popular Use |
|----------|------|--------------|-------------|
| JavaScript | Dynamic | 1995 | Web Development |
| Python | Dynamic | 1991 | Data Science, AI |
| Java | Static | 1995 | Enterprise Apps |
| TypeScript | Static | 2012 | Large-scale Web Apps |
| Go | Static | 2009 | Backend Services |

## Pricing Table

| Plan | Price | Features |
|------|-------|----------|
| Free | $0/mo | Basic features |
| Pro | $29/mo | Advanced features |
| Enterprise | Custom | All features + support |`;

const documentationMarkdown = `# API Documentation

## Introduction

This API allows you to interact with our service programmatically.

## Authentication

All API requests require authentication. Include your API key in the header:

\`\`\`bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com/endpoint
\`\`\`

## Endpoints

### GET /users

Retrieve a list of users.

**Parameters:**

- \`limit\` (optional): Maximum number of users to return
- \`offset\` (optional): Number of users to skip

**Response:**

\`\`\`json
{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "email": "john@example.com"
    }
  ],
  "total": 100
}
\`\`\`

### POST /users

Create a new user.

**Request Body:**

\`\`\`json
{
  "name": "Jane Doe",
  "email": "jane@example.com"
}
\`\`\`

## Rate Limiting

> **Important:** API requests are limited to 1000 requests per hour per API key.

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |`;

export const Basic: Story = {
    args: {
        children: basicMarkdown
    }
};

export const Comprehensive: Story = {
    args: {
        children: comprehensiveMarkdown
    }
};

export const CodeExamples: Story = {
    args: {
        children: codeExampleMarkdown
    }
};

export const Tables: Story = {
    args: {
        children: tableExampleMarkdown
    }
};

export const Documentation: Story = {
    args: {
        children: documentationMarkdown
    }
};

export const SmallSize: Story = {
    args: {
        children: comprehensiveMarkdown,
        size: "sm"
    }
};

export const LargeSize: Story = {
    args: {
        children: comprehensiveMarkdown,
        size: "lg"
    }
};

export const DarkTheme: Story = {
    args: {
        children: comprehensiveMarkdown,
        theme: "dark",
        className: "bg-slate-900 text-slate-100 p-6 rounded dark"
    }
};

export const CustomStyling: Story = {
    args: {
        children: basicMarkdown,
        className: "border border-slate-300 p-4 rounded-lg"
    }
};

// GitHub Flavored Markdown Features

const gfmFeaturesMarkdown = `# GitHub Flavored Markdown Features

## Strikethrough

You can ~~strikethrough~~ text using double tildes.

Example: ~~This text is incorrect~~ This is the correct text.

## Task Lists

### Project Tasks
- [x] Complete project setup
- [x] Install dependencies
- [ ] Write documentation
- [ ] Add unit tests
- [ ] Deploy to production

### Shopping List
- [x] Buy groceries
- [x] Get milk
- [ ] Buy bread
- [ ] Get fruits

## Autolinks

URLs are automatically converted to links:
https://github.com
www.example.com
user@example.com

## Tables with Alignment

| Left Aligned | Center Aligned | Right Aligned |
|:-------------|:--------------:|--------------:|
| Left         | Center         | Right         |
| Text         | Text           | Text          |
| 100          | 200            | 300           |`;

const syntaxHighlightingMarkdown = `# Code with Syntax Highlighting

## JavaScript with Full Highlighting

\`\`\`javascript
// React component example
import React, { useState, useEffect } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchUser() {
      try {
        const response = await fetch(\`/api/users/\${userId}\`);
        const data = await response.json();
        setUser(data);
      } catch (error) {
        console.error('Error fetching user:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchUser();
  }, [userId]);

  if (loading) return <div>Loading...</div>;
  if (!user) return <div>User not found</div>;

  return (
    <div className="user-profile">
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}

export default UserProfile;
\`\`\`

## Python with Syntax Highlighting

\`\`\`python
# Machine learning model example
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class MLModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.trained = False

    def train(self, X, y):
        """Train the model with features X and labels y"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.trained = True

        # Evaluate
        score = self.model.score(X_test, y_test)
        print(f"Model accuracy: {score:.2%}")
        return score

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)

# Usage
model = MLModel(n_estimators=200)
model.train(features, labels)
predictions = model.predict(new_data)
\`\`\`

## TypeScript with Highlighting

\`\`\`typescript
// Type definitions and interfaces
interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
}

type ApiResponse<T> = {
  data: T;
  status: number;
  message: string;
};

async function fetchUsers(): Promise<ApiResponse<User[]>> {
  try {
    const response = await fetch('/api/users');
    const data: ApiResponse<User[]> = await response.json();
    return data;
  } catch (error) {
    throw new Error(\`Failed to fetch users: \${error}\`);
  }
}

// Generic function with constraints
function findById<T extends { id: number }>(
  items: T[],
  id: number
): T | undefined {
  return items.find(item => item.id === id);
}
\`\`\`

## CSS with Highlighting

\`\`\`css
/* Modern CSS with custom properties */
:root {
  --primary-color: #3b82f6;
  --secondary-color: #8b5cf6;
  --spacing-unit: 8px;
}

.card {
  background: linear-gradient(
    135deg,
    var(--primary-color) 0%,
    var(--secondary-color) 100%
  );
  border-radius: calc(var(--spacing-unit) * 2);
  padding: calc(var(--spacing-unit) * 3);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease;
}

.card:hover {
  transform: translateY(-4px);
}

@media (max-width: 768px) {
  .card {
    padding: var(--spacing-unit);
  }
}
\`\`\``;

const imagesMarkdown = `# Markdown with Images

Images can be embedded using standard markdown syntax.

## Example Image

![React Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/200px-React-icon.svg.png)

## Image with Alt Text

![Placeholder Image](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/200px-React-icon.svg.png)

Images are automatically styled with rounded corners and proper spacing.`;

const allHeadingsMarkdown = `# Heading Level 1 - Main Title

This is a paragraph under H1. It demonstrates the largest heading size.

## Heading Level 2 - Section Title

This is a paragraph under H2. Perfect for major sections.

### Heading Level 3 - Subsection

This is a paragraph under H3. Great for subsections within a section.

#### Heading Level 4 - Minor Heading

This is a paragraph under H4. Used for smaller subdivisions.

##### Heading Level 5 - Small Heading

This is a paragraph under H5. Even smaller, for fine-grained organization.

###### Heading Level 6 - Smallest Heading

This is a paragraph under H6. The smallest heading available, displayed in uppercase.

---

All heading levels (H1-H6) are now fully supported and styled!`;

export const GithubFlavoredMarkdown: Story = {
    args: {
        children: gfmFeaturesMarkdown
    }
};

export const TaskLists: Story = {
    args: {
        children: `# Task Management

## Development Tasks
- [x] Set up project repository
- [x] Configure build tools
- [x] Create component structure
- [ ] Write unit tests
- [ ] Add integration tests
- [ ] Performance optimization
- [ ] Documentation review

## Code Review Checklist
- [x] Code follows style guide
- [x] All tests passing
- [ ] Documentation updated
- [ ] No console errors
- [ ] Accessibility checked`
    }
};

export const Strikethrough: Story = {
    args: {
        children: `# Text Formatting

## Strikethrough Examples

Regular text with ~~strikethrough~~ in the middle.

~~This entire line is struck through~~

**Bold text with ~~strikethrough~~** and *italic with ~~strikethrough~~*

## Common Use Cases

- ~~Old price: $99~~ New price: $79
- ~~Completed task~~
- ~~Deprecated method~~ Use newMethod() instead`
    }
};

export const WithSyntaxHighlighting: Story = {
    args: {
        children: syntaxHighlightingMarkdown,
        enableSyntaxHighlighting: true
    }
};

export const ImagesSupport: Story = {
    args: {
        children: imagesMarkdown
    }
};

export const AllHeadingLevels: Story = {
    args: {
        children: allHeadingsMarkdown
    }
};

export const CompleteFeatureShowcase: Story = {
    args: {
        children: `# Complete Markdown Feature Showcase

## Text Formatting

**Bold text**, *italic text*, ***bold and italic***, and ~~strikethrough~~.

## All Heading Levels

### H3 - Third Level
#### H4 - Fourth Level
##### H5 - Fifth Level
###### H6 - Sixth Level (Smallest)

## Lists

### Task List
- [x] Support basic markdown
- [x] Add GFM features
- [x] Implement syntax highlighting
- [ ] Add custom themes

### Regular Lists
1. First item
2. Second item
   - Nested bullet
   - Another nested item

## Code Examples

Inline code: \`const x = 42;\`

\`\`\`javascript
function greet(name) {
  return \`Hello, \${name}!\`;
}
\`\`\`

## Tables

| Feature | Status | Version |
|---------|--------|---------|
| Basic MD | ✅ | 1.0 |
| GFM | ✅ | 2.0 |
| Syntax | ✅ | 2.0 |

## Links & Images

Visit [GitHub](https://github.com) for more info.

> **Note:** This showcases all markdown features including GFM extensions!`,
        enableSyntaxHighlighting: true
    }
};

export const AdvancedFormatting: Story = {
    args: {
        children: `# Advanced Text Formatting

## Subscript and Superscript

- Water formula: H<sub>2</sub>O
- Einstein's equation: E = mc<sup>2</sup>
- Footnote reference<sup>1</sup>

## Keyboard Input

Press <kbd>Ctrl</kbd> + <kbd>C</kbd> to copy.

Use <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd> to open the command palette.

Common shortcuts:
- <kbd>Ctrl</kbd> + <kbd>S</kbd> - Save file
- <kbd>Ctrl</kbd> + <kbd>Z</kbd> - Undo
- <kbd>Ctrl</kbd> + <kbd>Y</kbd> - Redo
- <kbd>Alt</kbd> + <kbd>F4</kbd> - Close window

## Combined Formatting

You can combine **bold with <sup>superscript</sup>** or *italic with <sub>subscript</sub>*.

Mathematical expressions: x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> = r<sup>2</sup>

Chemical formulas: C<sub>6</sub>H<sub>12</sub>O<sub>6</sub> (Glucose)

## Technical Documentation

To install the package, run: <kbd>npm install react-markdown</kbd>

Navigate using arrow keys: <kbd>↑</kbd> <kbd>↓</kbd> <kbd>←</kbd> <kbd>→</kbd>`
    }
};

// Feature Demonstration Stories

const copyButtonDemo = `# Copy Button Feature

The copy button appears on code blocks, making it easy to copy code snippets.

\`\`\`javascript
// Hover over this code block to see the copy button
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

console.log(fibonacci(10)); // 55
\`\`\`

\`\`\`python
# Copy button also works with other languages
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
\`\`\`

Try clicking the copy button on any code block above!`;

const headingAnchorsDemo = `# Heading Anchors Feature

Hover over any heading to see the anchor link appear. Click the # symbol to get a direct link to that section.

## Introduction

This feature makes it easy to create table of contents and share links to specific sections.

### Benefits

- Direct linking to sections
- Easy navigation
- Better user experience

#### Technical Implementation

The anchors are automatically generated from the heading text.

##### Slug Generation

Special characters are removed and spaces are converted to hyphens.

###### Example Output

Hover over each heading above to see the anchor link!`;

const alertsDemo = `# Alert/Callout Boxes

GitHub-style alerts for highlighting important information.

> [!NOTE]
> This is a note alert. Use it for neutral or informational content.

> [!TIP]
> This is a tip alert. Great for helpful suggestions and best practices!

> [!IMPORTANT]
> This is an important alert. Use it to draw attention to key information.

> [!WARNING]
> This is a warning alert. Perfect for cautionary information that users should be aware of.

> [!CAUTION]
> This is a caution alert. Use for critical warnings about dangerous actions or consequences.

## Using Alerts

Simply use the blockquote syntax with \`[!TYPE]\` at the beginning:

\`\`\`markdown
> [!NOTE]
> Your note content here
\`\`\`

Available types: NOTE, TIP, IMPORTANT, WARNING, CAUTION`;

const lineNumbersDemo = `# Line Numbers Feature

Code blocks can display line numbers for easier reference.

\`\`\`javascript
// This example shows line numbers
function calculateTotal(items) {
  return items.reduce((sum, item) => {
    return sum + (item.price * item.quantity);
  }, 0);
}

const cartItems = [
  { name: "Widget", price: 10, quantity: 2 },
  { name: "Gadget", price: 15, quantity: 1 },
  { name: "Doohickey", price: 5, quantity: 3 }
];

const total = calculateTotal(cartItems);
console.log(\`Total: $\${total}\`); // Total: $50
\`\`\`

\`\`\`python
# Line numbers make it easy to reference specific lines
class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def calculate_total(self):
        return sum(item['price'] * item['quantity']
                   for item in self.items)

    def checkout(self):
        total = self.calculate_total()
        print(f"Total: \${total}")
        self.items = []
\`\`\``;

export const WithCopyButton: Story = {
    args: {
        children: copyButtonDemo,
        enableSyntaxHighlighting: true,
        enableCopyButton: true
    }
};

export const WithHeadingAnchors: Story = {
    args: {
        children: headingAnchorsDemo,
        enableHeadingAnchors: true
    }
};

export const WithAlerts: Story = {
    args: {
        children: alertsDemo
    }
};

export const WithLineNumbers: Story = {
    args: {
        children: lineNumbersDemo,
        enableSyntaxHighlighting: true,
        enableLineNumbers: true
    }
};
