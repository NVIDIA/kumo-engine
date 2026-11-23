#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <arrow/c/abi.h>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// The GIL stays held: PyOS_double_to_string provides exactly CPython's JSON
// float spelling without constructing a Python float for each Arrow value.
struct PythonError {};
struct Writer {
    std::string data;
    size_t limit;
    explicit Writer(size_t limit) : limit(limit) {}
    void append(const char* s, size_t n) {
        if (n > limit - data.size())
            throw std::invalid_argument("Payload exceeds byte budget; reduce the workload batch");
        data.append(s, n);
    }
    void append(const char* s) { append(s, std::strlen(s)); }
    void put(char c) { append(&c, 1); }
    void integer(int64_t n) {
        char buf[32];
        auto end = std::to_chars(buf, buf + sizeof(buf), n).ptr;
        append(buf, static_cast<size_t>(end - buf));
    }
    void hex(uint32_t cp) {
        static constexpr char digits[] = "0123456789abcdef";
        char buf[] = {'\\', 'u', digits[(cp >> 12) & 15], digits[(cp >> 8) & 15],
                      digits[(cp >> 4) & 15], digits[cp & 15]};
        append(buf, sizeof(buf));
    }
    // Match json.dumps(ensure_ascii=True), including supplementary pairs.
    void string(const char* s, int64_t n) {
        put('"');
        int64_t i = 0;
        while (i < n) {
            int64_t begin = i;
            while (i < n && static_cast<uint8_t>(s[i]) >= 0x20 &&
                   static_cast<uint8_t>(s[i]) < 0x7f && s[i] != '"' && s[i] != '\\') ++i;
            append(s + begin, static_cast<size_t>(i - begin));
            if (i == n) break;
            uint32_t cp = static_cast<uint8_t>(s[i++]);
            switch (cp) {
                case '"': append("\\\""); continue;
                case '\\': append("\\\\"); continue;
                case '\b': append("\\b"); continue;
                case '\f': append("\\f"); continue;
                case '\n': append("\\n"); continue;
                case '\r': append("\\r"); continue;
                case '\t': append("\\t"); continue;
            }
            if (cp < 0x80) { hex(cp); continue; }
            int continuation;
            uint32_t minimum;
            if (cp >= 0xc2 && cp <= 0xdf) { continuation = 1; minimum = 0x80; cp &= 31; }
            else if (cp >= 0xe0 && cp <= 0xef) { continuation = 2; minimum = 0x800; cp &= 15; }
            else if (cp >= 0xf0 && cp <= 0xf4) { continuation = 3; minimum = 0x10000; cp &= 7; }
            else throw std::invalid_argument("Invalid UTF-8 in Arrow string");
            if (continuation > n - i) throw std::invalid_argument("Truncated UTF-8 in Arrow string");
            for (int j = 0; j < continuation; ++j) {
                uint8_t byte = static_cast<uint8_t>(s[i++]);
                if ((byte & 0xc0) != 0x80) throw std::invalid_argument("Invalid UTF-8 continuation");
                cp = (cp << 6) | (byte & 63);
            }
            if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
                throw std::invalid_argument("Invalid Unicode code point in Arrow string");
            if (cp <= 0xffff) hex(cp);
            else { cp -= 0x10000; hex(0xd800 + (cp >> 10)); hex(0xdc00 + (cp & 1023)); }
        }
        put('"');
    }
    void temporal(int64_t value, bool date) {
        int64_t seconds, micros = 0;
        if (date) seconds = value * 86400;
        else {
            seconds = value / 1000000;
            micros = value % 1000000;
            if (micros < 0) { micros += 1000000; --seconds; }
        }
        time_t raw = static_cast<time_t>(seconds);
        struct tm parts {};
        #ifdef _WIN32
        // Windows' calendar routine rejects dates before 1970 and after 3000.
        // Gregorian dates repeat every 400 years; convert inside that range.
        constexpr int64_t era_seconds = 146097LL * 86400;
        int64_t eras = seconds / era_seconds;
        int64_t within_era = seconds % era_seconds;
        if (within_era < 0) { within_era += era_seconds; --eras; }
        time_t shifted = static_cast<time_t>(within_era);
        bool converted = gmtime_s(&parts, &shifted) == 0;
        parts.tm_year += static_cast<int>(eras * 400);
        #else
        bool converted = gmtime_r(&raw, &parts) != nullptr;
        #endif
        if (static_cast<int64_t>(raw) != seconds || !converted)
            throw std::invalid_argument("Arrow timestamp outside platform calendar range");
        int64_t year = static_cast<int64_t>(parts.tm_year) + 1900;
        put('"');
        if (year < 0) { put('-'); year = -year; }
        if (year < 1000) put('0');
        if (year < 100) put('0');
        if (year < 10) put('0');
        integer(year);
        char buf[32];
        int n = PyOS_snprintf(buf, sizeof(buf), "-%02d-%02d", parts.tm_mon + 1, parts.tm_mday);
        append(buf, n);
        if (!date) {
            n = PyOS_snprintf(buf, sizeof(buf), "T%02d:%02d:%02d.%06dZ",
                              parts.tm_hour, parts.tm_min, parts.tm_sec, static_cast<int>(micros));
            append(buf, n);
        }
        put('"');
    }
};

enum class Type { Int64, Double, Bool, String, Date, Timestamp };
Type type_of(const ArrowSchema* s) {
    if (!s || !s->format || s->dictionary || s->n_children)
        throw std::invalid_argument("Native arrays encoder requires normalized scalar Arrow columns");
    const char* f = s->format;
    if (!std::strcmp(f, "l")) return Type::Int64;
    if (!std::strcmp(f, "g")) return Type::Double;
    if (!std::strcmp(f, "b")) return Type::Bool;
    if (!std::strcmp(f, "u")) return Type::String;
    if (!std::strcmp(f, "tdD")) return Type::Date;
    if (!std::strcmp(f, "tsu:UTC")) return Type::Timestamp;
    throw std::invalid_argument(std::string("Unsupported native arrays Arrow format: ") + f);
}

bool bit(const void* buffer, int64_t index) {
    return (static_cast<const uint8_t*>(buffer)[index >> 3] >> (index & 7)) & 1;
}
void cell(Writer& out, Type type, const ArrowArray* a, int64_t row) {
    int64_t i = a->offset + row;
    if (a->buffers[0] && !bit(a->buffers[0], i)) { out.append("null"); return; }
    switch (type) {
        case Type::Int64: {
            int64_t v = static_cast<const int64_t*>(a->buffers[1])[i];
            bool quote = v < -9007199254740991LL || v > 9007199254740991LL;
            if (quote) out.put('"');
            out.integer(v);
            if (quote) out.put('"');
            break;
        }
        case Type::Double: {
            double v = static_cast<const double*>(a->buffers[1])[i];
            if (!std::isfinite(v)) throw std::invalid_argument("Out of range float values are not JSON compliant");
            char* text = PyOS_double_to_string(v, 'r', 0, Py_DTSF_ADD_DOT_0, nullptr);
            if (!text) throw PythonError{};
            try { out.append(text); } catch (...) { PyMem_Free(text); throw; }
            PyMem_Free(text);
            break;
        }
        case Type::Bool: out.append(bit(a->buffers[1], i) ? "true" : "false"); break;
        case Type::String: {
            const auto* offsets = static_cast<const int32_t*>(a->buffers[1]);
            int64_t start = offsets[i], end = offsets[i + 1];
            if (start < 0 || end < start) throw std::invalid_argument("Invalid Arrow string offsets");
            // Empty buffers may legally have a null data pointer.
            const char* bytes = end == start ? "" : static_cast<const char*>(a->buffers[2]) + start;
            out.string(bytes, end - start);
            break;
        }
        case Type::Date: out.temporal(static_cast<const int32_t*>(a->buffers[1])[i], true); break;
        case Type::Timestamp: out.temporal(static_cast<const int64_t*>(a->buffers[1])[i], false); break;
    }
}
struct SchemaOwner {
    ArrowSchema value {};
    ~SchemaOwner() { if (value.release) value.release(&value); }
};
struct ArrayOwner {
    ArrowArray value {};
    ~ArrayOwner() { if (value.release) value.release(&value); }
};
void stream_error(ArrowArrayStream* stream) {
    const char* error = stream->get_last_error(stream);
    throw std::invalid_argument(error ? error : "Arrow stream failed");
}
PyObject* encode_rows(PyObject*, PyObject* args) {
    PyObject* capsule;
    Py_ssize_t limit;
    if (!PyArg_ParseTuple(args, "On:encode_rows", &capsule, &limit)) return nullptr;
    if (limit < 0) { PyErr_SetString(PyExc_ValueError, "Negative payload byte budget"); return nullptr; }
    auto* stream = static_cast<ArrowArrayStream*>(PyCapsule_GetPointer(capsule, "arrow_array_stream"));
    if (!stream) return nullptr;
    if (!stream->release || !stream->get_schema || !stream->get_next || !stream->get_last_error) {
        PyErr_SetString(PyExc_ValueError, "Released or invalid Arrow stream"); return nullptr;
    }
    // Borrow the stream for this synchronous call. Its PyCapsule owns/releases
    // it; each schema and batch obtained here has its own release callback.
    try {
        SchemaOwner schema;
        if (stream->get_schema(stream, &schema.value)) stream_error(stream);
        const auto& s = schema.value;
        if (!s.release || !s.format || std::strcmp(s.format, "+s") || s.dictionary || s.n_children < 0)
            throw std::invalid_argument("Native arrays encoder requires an Arrow table stream");
        std::vector<Type> types;
        for (int64_t j = 0; j < s.n_children; ++j) types.push_back(type_of(s.children[j]));
        Writer out(static_cast<size_t>(limit));
        out.put('[');
        bool first = true;
        while (true) {
            ArrayOwner batch;
            if (stream->get_next(stream, &batch.value)) stream_error(stream);
            const auto& a = batch.value;
            if (!a.release) break;
            if (a.length < 0 || a.offset < 0 || a.n_children != s.n_children || a.null_count != 0)
                throw std::invalid_argument("Invalid Arrow record batch");
            for (int64_t j = 0; j < a.n_children; ++j) {
                const auto* child = a.children[j];
                int needed = types[j] == Type::String ? 3 : 2;
                if (!child || child->offset < 0 || child->length < a.length + a.offset ||
                    !child->buffers || child->n_buffers != needed)
                    throw std::invalid_argument("Invalid Arrow column layout");
            }
            // Match zip(*columns) for the degenerate zero-column table.
            for (int64_t row = 0; row < a.length && !types.empty(); ++row) {
                if ((row & 4095) == 0 && PyErr_CheckSignals() < 0) throw PythonError{};
                if (!first) out.put(',');
                first = false;
                out.put('[');
                for (int64_t j = 0; j < a.n_children; ++j) {
                    if (j) out.put(',');
                    cell(out, types[j], a.children[j], row + a.offset);
                }
                out.put(']');
            }
        }
        out.put(']');
        return PyBytes_FromStringAndSize(out.data.data(), static_cast<Py_ssize_t>(out.data.size()));
    } catch (const PythonError&) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
    } catch (const std::bad_alloc&) { PyErr_NoMemory(); }
      catch (const std::exception& e) { PyErr_SetString(PyExc_ValueError, e.what()); }
    return nullptr;
}
PyMethodDef methods[] = {{"encode_rows", encode_rows, METH_VARARGS,
                         "Encode normalized Arrow C stream rows as compact ASCII JSON bytes."},
                        {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native_json", nullptr, -1, methods};
} // namespace
PyMODINIT_FUNC PyInit__native_json() { return PyModule_Create(&module); }
