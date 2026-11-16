## Stacktracer

Tool for mapping minified stack trace to actual source file and location. Production build will generate the source map file and relocates it to source_map folder inside the directory(`kumo/ui/stacktracer/`).

## Installation

Run the following command inside the directory(`kumo/ui/stacktracer/`) to install required npm packages:

```bash
 npm install
```

## Resolve the minified stack trace

Run the below command to map the minified stack trace to actual source file with location.

-   Replace `<source map file>` in below command with the actual source map file inside the source_map directory.
-   `<line>` and `<column>` in below command with the line and column from minified stack trace.
    ```bash
     npm run trace source_map/<source map file> <line> <column>
    ```

## Usage

Search the configured backend logs for the UI error. The log starts with something like "Received error from UI:". Next there will be one line description of the error message and then stacktrace with location (line and column) in the UI code that raised the error.

```
$ npm run trace source_map/main.b4a27605.js.map 2 2344107
Maps to pages/connector/Connectors.js:77:66 (name)

              connector.name = response.data[i].configs.name;
                                                        ^

$ npm run trace source_map/main.7aca32e4.js.map 2 2897180
Maps to pages/tables/Table.js:154:58 (table_name)

               table.name = response.data[i].configs.table_name;
                                                     ^
```
