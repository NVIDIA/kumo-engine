// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import http, {
    type IncomingHttpHeaders,
    type IncomingMessage,
    type Server,
    type ServerResponse
} from "node:http";
import { type AddressInfo } from "node:net";
import { createServer, type ViteDevServer } from "vite";
import { afterEach, describe, expect, it } from "vitest";

import { createLocalApiProxy } from "./viteApiProxy";

type HttpResult = {
    body: string;
    headers: IncomingHttpHeaders;
    status: number;
};

const listen = async (server: Server) => {
    await new Promise<void>((resolve, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", () => resolve());
    });
    return (server.address() as AddressInfo).port;
};

const request = (
    port: number,
    path: string,
    headers: Record<string, string> = {}
) =>
    new Promise<HttpResult>((resolve, reject) => {
        const outgoing = http.request(
            {
                host: "127.0.0.1",
                port,
                path,
                headers: { host: `localhost:${port}`, ...headers }
            },
            (response) => {
                const chunks: Buffer[] = [];
                response.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
                response.on("end", () =>
                    resolve({
                        body: Buffer.concat(chunks).toString("utf8"),
                        headers: response.headers,
                        status: response.statusCode || 0
                    })
                );
            }
        );
        outgoing.once("error", reject);
        outgoing.end();
    });

describe("createLocalApiProxy", () => {
    it("does not change deployed builds", () => {
        expect(createLocalApiProxy({})).toBeUndefined();
        expect(createLocalApiProxy({ UI_LOCAL: "false" })).toBeUndefined();
    });

    it("requires an explicit upstream origin", () => {
        expect(() =>
            createLocalApiProxy({ UI_LOCAL: "true", REACT_APP_PORT: "12000" })
        ).toThrow(/API_PROXY_TARGET is required/);
    });

    it.each([
        "ftp://127.0.0.1:12001",
        "http://user:secret@127.0.0.1:12001",
        "http://127.0.0.1:12001/private",
        "http://127.0.0.1:12001/?token=secret",
        "not a url"
    ])("rejects unsafe or ambiguous upstream target %s", (target) => {
        expect(() =>
            createLocalApiProxy({
                UI_LOCAL: "true",
                API_PROXY_TARGET: target
            })
        ).toThrow(/API_PROXY_TARGET/);
    });

    it.each(["API_TARGET", "REACT_APP_BASE_URL_DEV"])(
        "rejects a split-origin %s and reports the canonical callback",
        (name) => {
            expect(() =>
                createLocalApiProxy({
                    UI_LOCAL: "true",
                    API_PROXY_TARGET: "http://127.0.0.1:12001",
                    REACT_APP_PORT: "12000",
                    [name]: "http://127.0.0.1:12001"
                })
            ).toThrow(/http:\/\/localhost:12000\/v0\/auth\/callback/);
        }
    );

    it("derives both internal API proxies from the configured target", () => {
        const proxy = createLocalApiProxy({
            UI_LOCAL: "yes",
            API_PROXY_TARGET: "http://127.0.0.1:12001/",
            REACT_APP_BASE_URL_DEV: "/",
            REACT_APP_PORT: "12000"
        });

        expect(Object.keys(proxy || {})).toEqual(["/v0", "/v1"]);
        expect(proxy?.["/v0"]).toMatchObject({
            target: "http://127.0.0.1:12001",
            changeOrigin: false
        });
        expect(proxy?.["/v1"]).toMatchObject({
            target: "http://127.0.0.1:12001",
            changeOrigin: false
        });
    });
});

describe("local API proxy integration", () => {
    let backend: Server | undefined;
    let vite: ViteDevServer | undefined;
    const seen: Array<{
        headers: IncomingHttpHeaders;
        url: string;
    }> = [];

    afterEach(async () => {
        await vite?.close();
        if (backend?.listening) {
            await new Promise<void>((resolve, reject) =>
                backend?.close((error) => (error ? reject(error) : resolve()))
            );
        }
        backend = undefined;
        vite = undefined;
        seen.length = 0;
    });

    it("preserves browser Host, query, redirects, and cookies across v0/v1", async () => {
        backend = http.createServer(
            (incoming: IncomingMessage, response: ServerResponse) => {
                seen.push({
                    headers: incoming.headers,
                    url: incoming.url || ""
                });
                if (incoming.url?.startsWith("/v0/auth/callback")) {
                    response.writeHead(307, {
                        location: "/home",
                        "set-cookie":
                            "kumo_direct_connect_session=session-id; " +
                            "HttpOnly; SameSite=Lax; Path=/"
                    });
                    response.end();
                    return;
                }
                if (incoming.url === "/v0/auth/me") {
                    response.writeHead(
                        incoming.headers.cookie?.includes("session-id")
                            ? 200
                            : 401,
                        { "content-type": "application/json" }
                    );
                    response.end('{"user_id":"user-1"}');
                    return;
                }
                response.writeHead(200, { "content-type": "text/plain" });
                response.end(incoming.url);
            }
        );
        const backendPort = await listen(backend);
        const proxy = createLocalApiProxy({
            UI_LOCAL: "true",
            API_PROXY_TARGET: `http://127.0.0.1:${backendPort}`,
            REACT_APP_BASE_URL_DEV: "/",
            REACT_APP_PORT: "12000"
        });
        vite = await createServer({
            configFile: false,
            logLevel: "silent",
            server: {
                host: "127.0.0.1",
                port: 0,
                strictPort: false,
                proxy
            }
        });
        await vite.listen();
        const vitePort = (vite.httpServer?.address() as AddressInfo).port;

        const login = await request(
            vitePort,
            "/v0/auth/login?return_to=%2Fgraphs%2F%3Fproject%3Ddemo",
            {
                forwarded: "host=evil.example;proto=https",
                "x-forwarded-for": "203.0.113.10",
                "x-forwarded-host": "evil.example",
                "x-forwarded-port": "443",
                "x-forwarded-proto": "https"
            }
        );
        expect(login.status).toBe(200);
        expect(login.body).toBe(
            "/v0/auth/login?return_to=%2Fgraphs%2F%3Fproject%3Ddemo"
        );
        expect(seen[0].headers.host).toBe(`localhost:${vitePort}`);
        expect(seen[0].headers.forwarded).toBeUndefined();
        expect(seen[0].headers["x-forwarded-for"]).toBeUndefined();
        expect(seen[0].headers["x-forwarded-host"]).toBeUndefined();
        expect(seen[0].headers["x-forwarded-port"]).toBeUndefined();
        expect(seen[0].headers["x-forwarded-proto"]).toBeUndefined();

        const callback = await request(
            vitePort,
            "/v0/auth/callback?code=a%2Fb%3Dc&state=state-1"
        );
        expect(callback.status).toBe(307);
        expect(callback.headers.location).toBe("/home");
        expect(callback.headers["set-cookie"]?.[0]).toContain("HttpOnly");
        expect(seen[1].url).toBe(
            "/v0/auth/callback?code=a%2Fb%3Dc&state=state-1"
        );

        const me = await request(vitePort, "/v0/auth/me", {
            cookie: callback.headers["set-cookie"]?.[0].split(";", 1)[0] || ""
        });
        expect(me.status).toBe(200);
        expect(me.body).toBe('{"user_id":"user-1"}');

        const v1 = await request(vitePort, "/v1/projects?limit=10");
        expect(v1.status).toBe(200);
        expect(v1.body).toBe("/v1/projects?limit=10");
    });
});
