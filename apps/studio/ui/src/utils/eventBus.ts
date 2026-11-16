// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Wrapper for document event handler/listener
 * This wrapper can be accessed by using eventBus
 *
 * Example :
 *   - eventBus.on(<event_name>);
 *
 * on :
 *   - Appends an event listener for events.
 *   - The callback argument sets the callback that will be invoked when the
 *   event is dispatched.
 *
 * dispatch :
 *   - Dispatches a event
 *   - Specified event listener will recieves the callback
 *
 * remove :
 *   - Removes the specified event
 *   - Detaches the listener that `on` registered for this callback
 **/

// Define types for event names and callback functions
type EventCallback<T> = (data: T) => void;
type EventName = string;

// `on` wraps each callback in a DOM EventListener (to unwrap `detail`). To make
// `remove` work, remember the wrapper keyed by the original callback so the
// exact listener that was added can be detached again.
const handlerRegistry = new Map<
    EventName,
    Map<EventCallback<never>, EventListener>
>();

const eventBus = {
    on<T>(event: EventName, callback: EventCallback<T>): void {
        let handlers = handlerRegistry.get(event);
        if (!handlers) {
            handlers = new Map();
            handlerRegistry.set(event, handlers);
        }
        // Idempotent per (event, callback): registering the same callback again
        // is a no-op (mirrors addEventListener for a same-reference handler) so
        // a second listener can't leak past remove().
        if (handlers.has(callback as EventCallback<never>)) return;

        const wrapped = ((e: Event) =>
            callback((e as CustomEvent<T>).detail)) as EventListener;
        handlers.set(callback as EventCallback<never>, wrapped);

        document.addEventListener(event, wrapped);
    },
    dispatch<T>(event: EventName, data?: T): void {
        document.dispatchEvent(new CustomEvent(event, { detail: data }));
    },
    remove<T>(event: EventName, callback: EventCallback<T>): void {
        const handlers = handlerRegistry.get(event);
        if (!handlers) return;
        const wrapped = handlers.get(callback as EventCallback<never>);
        if (!wrapped) return;

        document.removeEventListener(event, wrapped);
        handlers.delete(callback as EventCallback<never>);
        if (handlers.size === 0) {
            handlerRegistry.delete(event);
        }
    }
};

export default eventBus;
