// proxy_ort.cpp -- onnxruntime.dll shim used ONLY for one-time model extraction.
//
// oneocr.dll imports onnxruntime.dll by ordinal 1 (OrtGetApiBase). This shim
// exports exactly that entry point, loads the genuine runtime from
// onnxruntime_real.dll, clones its OrtApi table and replaces the
// CreateSessionFromArray* members so every decrypted model blob handed to ORT
// is dumped to disk before inference.
//
// Build: see build.ps1 (x64, static CRT).
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#define OrtGetApiBase OrtGetApiBase_decl_only
#include "onnxruntime_c_api.h"
#undef OrtGetApiBase

typedef const OrtApiBase*(ORT_API_CALL* OrtGetApiBaseFn)(void);

static HMODULE     g_real_module = nullptr;
static OrtApiBase* g_real_base   = nullptr;
static OrtApi      g_api;                 // clone of the real table
static OrtApiBase  g_fake_base;           // what we hand back to oneocr.dll
static decltype(g_api.CreateSessionFromArray) g_orig_csfa = nullptr;
static decltype(g_api.CreateSessionFromArrayWithPrepackedWeightsContainer) g_orig_csfa_pw = nullptr;
static LONG        g_seq = 0;
static bool        g_ready = false;
static CRITICAL_SECTION g_lock;

static void ensure_lock() {
    static LONG once = 0;
    if (InterlockedCompareExchange(&once, 1, 0) == 0) {
        InitializeCriticalSection(&g_lock);
    }
}

static void dump_dir(wchar_t* out, DWORD cch) {
    DWORD n = GetEnvironmentVariableW(L"ONEOCR_DUMP_DIR", out, cch);
    if (n == 0 || n >= cch) {
        GetCurrentDirectoryW(cch, out);
        wcscat_s(out, cch, L"\\oneocr_dump");
    }
    CreateDirectoryW(out, nullptr);
}

static void log_line(const char* fmt, ...) {
    wchar_t dir[MAX_PATH]; dump_dir(dir, MAX_PATH);
    wchar_t path[MAX_PATH];
    swprintf_s(path, L"%ls\\proxy_log.txt", dir);
    FILE* f = nullptr;
    if (_wfopen_s(&f, path, L"a") == 0 && f) {
        va_list ap; va_start(ap, fmt);
        vfprintf(f, fmt, ap);
        va_end(ap);
        fputc('\n', f);
        fclose(f);
    }
}

static LONG dump_blob(const char* tag, const void* data, size_t len) {
    ensure_lock();
    EnterCriticalSection(&g_lock);
    LONG idx = InterlockedIncrement(&g_seq);
    LeaveCriticalSection(&g_lock);

    wchar_t dir[MAX_PATH]; dump_dir(dir, MAX_PATH);
    wchar_t path[MAX_PATH];
    swprintf_s(path, L"%ls\\model_%03ld_%llu.bin", dir, idx, (unsigned long long)len);
    FILE* f = nullptr;
    if (_wfopen_s(&f, path, L"wb") == 0 && f) {
        size_t written = fwrite(data, 1, len, f);
        fclose(f);
        const unsigned char* p = (const unsigned char*)data;
        log_line("[dump] #%ld %s len=%llu wrote=%llu magic=%02X%02X%02X%02X",
                 idx, tag, (unsigned long long)len, (unsigned long long)written,
                 len > 3 ? p[0] : 0, len > 3 ? p[1] : 0,
                 len > 3 ? p[2] : 0, len > 3 ? p[3] : 0);
    } else {
        log_line("[dump] #%ld FAILED to open output file", idx);
    }
    return idx;
}

// ---------------------------------------------------------------- IO tracing
// When ONEOCR_DUMP_IO=1, every OrtApi::Run call writes the model index plus all
// input/output tensors (header JSON + raw payload) into <dump_dir>\io\.
static decltype(g_api.Run) g_orig_run = nullptr;
static CRITICAL_SECTION g_io_lock;
static LONG g_run_seq = 0;
static int  g_sess_model[128];
static OrtSession* g_sess_ptr[128];
static int  g_sess_n = 0;

static bool io_enabled() {
    wchar_t v[8];
    DWORD n = GetEnvironmentVariableW(L"ONEOCR_DUMP_IO", v, 8);
    return n > 0 && v[0] == L'1';
}

static int model_of_session(OrtSession* s) {
    for (int i = 0; i < g_sess_n; ++i)
        if (g_sess_ptr[i] == s) return g_sess_model[i];
    return -1;
}

static void register_session(OrtSession* s, int model_idx) {
    EnterCriticalSection(&g_io_lock);
    if (g_sess_n < 128) { g_sess_ptr[g_sess_n] = s; g_sess_model[g_sess_n] = model_idx; ++g_sess_n; }
    LeaveCriticalSection(&g_io_lock);
}

static size_t elem_size(int dt) {
    switch (dt) {
        case 1: case 6: case 12: return 4;      // float, int32, uint32
        case 2: case 3: case 9: return 1;       // uint8, int8, bool
        case 4: case 5: case 10: return 2;      // uint16, int16, float16
        case 7: case 13: return 8;              // int64, uint64
        case 11: return 8;                      // double
        default: return 0;
    }
}

static void dump_tensor(const wchar_t* dir, int run_id, const char* tag, const char* name,
                        const OrtValue* val, int model_idx) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (g_api.GetTensorTypeAndShape(val, &info) != nullptr || !info) return;
    size_t nd = 0; g_api.GetDimensionsCount(info, &nd);
    int64_t dims[8] = {0};
    if (nd > 8) nd = 8;
    g_api.GetDimensions(info, dims, nd);
    size_t total = 0; g_api.GetTensorShapeElementCount(info, &total);
    ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    g_api.GetTensorElementType(info, &dt);
    size_t es = elem_size((int)dt);
    void* raw = nullptr; g_api.GetTensorMutableData(const_cast<OrtValue*>(val), &raw);
    g_api.ReleaseTensorTypeAndShapeInfo(info);
    if (!raw || !es) return;

    size_t keep = total > 200000 ? 200000 : total;
    wchar_t path[MAX_PATH];
    swprintf_s(path, L"%ls\\io\\r%03d_m%d_%s_%hs.bin", dir, run_id, model_idx, tag, name);
    FILE* f = nullptr;
    if (_wfopen_s(&f, path, L"wb") != 0 || !f) return;
    fprintf(f, "{\"model\":%d,\"dtype\":%d,\"total\":%llu,\"kept\":%llu,\"shape\":[",
            model_idx, (int)dt, (unsigned long long)total, (unsigned long long)keep);
    for (size_t i = 0; i < nd; ++i) fprintf(f, "%s%lld", i ? "," : "", (long long)dims[i]);
    fprintf(f, "]}\n");
    fwrite(raw, es, keep, f);
    fclose(f);
}

static decltype(g_api.RunWithBinding) g_orig_runwb = nullptr;
static decltype(g_api.RunAsync) g_orig_runasync = nullptr;

static OrtStatus* ORT_API_CALL Hook_RunWithBinding(
        OrtSession* sess, const OrtRunOptions* ro, const OrtIoBinding* binding_ptr) NO_EXCEPTION {
    static LONG c = 0;
    if (InterlockedIncrement(&c) <= 3) log_line("[probe] RunWithBinding called");
    OrtStatus* st = g_orig_runwb(sess, ro, binding_ptr);
    return st;
}

static OrtStatus* ORT_API_CALL Hook_RunAsync(
        OrtSession* sess, const OrtRunOptions* ro,
        const char* const* input_names, const OrtValue* const* inputs, size_t input_len,
        const char* const* output_names, size_t output_names_len,
        OrtValue** output, RunAsyncCallbackFn callback, void* user_data) NO_EXCEPTION {
    static LONG c = 0;
    if (InterlockedIncrement(&c) <= 3) log_line("[probe] RunAsync called");
    return g_orig_runasync(sess, ro, input_names, inputs, input_len, output_names, output_names_len, output, callback, user_data);
}
static decltype(g_api.CreateTensorWithDataAsOrtValue) g_orig_ctwda = nullptr;

static OrtStatus* ORT_API_CALL Hook_CreateTensorWithDataAsOrtValue(
        const OrtMemoryInfo* info, void* p_data, size_t p_data_len,
        const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
        OrtValue** out) NO_EXCEPTION {
    OrtStatus* st = g_orig_ctwda(info, p_data, p_data_len, shape, shape_len, type, out);
    if (io_enabled()) {
        wchar_t dir[MAX_PATH]; dump_dir(dir, MAX_PATH);
        wchar_t iodir[MAX_PATH]; swprintf_s(iodir, L"%ls\\io", dir);
        CreateDirectoryW(iodir, nullptr);
        static LONG c = 0;
        int id = (int)InterlockedIncrement(&c) - 1;
        wchar_t path[MAX_PATH];
        swprintf_s(path, L"%ls\\tensor_%03d.bin", iodir, id);
        FILE* f = nullptr;
        if (_wfopen_s(&f, path, L"wb") == 0 && f) {
            fprintf(f, "{\"dtype\":%d,\"bytes\":%llu,\"shape\":[", (int)type, (unsigned long long)p_data_len);
            for (size_t i = 0; i < shape_len; ++i) fprintf(f, "%s%lld", i ? "," : "", (long long)shape[i]);
            fprintf(f, "]}\n");
            fwrite(p_data, 1, p_data_len, f);
            fclose(f);
        }
        log_line("[tensor] #%d dtype=%d bytes=%llu ndim=%zu", id, (int)type,
                 (unsigned long long)p_data_len, shape_len);
    }
    return st;
}
static OrtStatus* ORT_API_CALL Hook_Run(
        OrtSession* sess, const OrtRunOptions* ro,
        const char* const* input_names, const OrtValue* const* inputs, size_t input_len,
        const char* const* output_names, size_t output_len, OrtValue** outputs) NO_EXCEPTION {
    OrtStatus* st = g_orig_run(sess, ro, input_names, inputs, input_len, output_names, output_len, outputs);
    if (!io_enabled() || g_api.GetTensorTypeAndShape == nullptr) return st;

    wchar_t dir[MAX_PATH]; dump_dir(dir, MAX_PATH);
    wchar_t iodir[MAX_PATH]; swprintf_s(iodir, L"%ls\\io", dir);
    CreateDirectoryW(iodir, nullptr);

    EnterCriticalSection(&g_io_lock);
    int run_id = (int)InterlockedIncrement(&g_run_seq) - 1;
    LeaveCriticalSection(&g_io_lock);

    int midx = model_of_session(sess);
    for (size_t i = 0; i < input_len; ++i)
        dump_tensor(iodir, run_id, "in", input_names && input_names[i] ? input_names[i] : "x", inputs[i], midx);
    if (st == nullptr) {
        for (size_t i = 0; i < output_len; ++i)
            dump_tensor(iodir, run_id, "out", output_names && output_names[i] ? output_names[i] : "y", outputs[i], midx);
    }
    return st;
}
static OrtStatusPtr ORT_API_CALL Hook_CreateSessionFromArray(
        const OrtEnv* env, const void* model_data, size_t model_data_length,
        const OrtSessionOptions* options, OrtSession** out) NO_EXCEPTION {
    LONG idx = dump_blob("CreateSessionFromArray", model_data, model_data_length);
    OrtStatus* st = g_orig_csfa(env, model_data, model_data_length, options, out);
    if (st == nullptr && out && *out) register_session(*out, (int)idx);
    return st;
}

static OrtStatusPtr ORT_API_CALL Hook_CreateSessionFromArrayWithPrepackedWeightsContainer(
        const OrtEnv* env, const void* model_data, size_t model_data_length,
        const OrtSessionOptions* options, OrtPrepackedWeightsContainer* container,
        OrtSession** out) NO_EXCEPTION {
    LONG idx = dump_blob("CreateSessionFromArrayWithPrepackedWeightsContainer", model_data, model_data_length);
    OrtStatus* st = g_orig_csfa_pw(env, model_data, model_data_length, options, container, out);
    if (st == nullptr && out && *out) register_session(*out, (int)idx);
    return st;
}

static const char* ORT_API_CALL Hook_GetVersionString(void) NO_EXCEPTION {
    static char buf[64] = {0};
    if (!buf[0] && g_real_base && g_real_base->GetVersionString) {
        strncpy_s(buf, g_real_base->GetVersionString(), _TRUNCATE);
    }
    return buf;
}

static const OrtApi* ORT_API_CALL Hook_GetApi(uint32_t version) NO_EXCEPTION {
    (void)version;
    return &g_api;
}

extern "C" __declspec(dllexport) const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION {
    ensure_lock();
    if (!g_ready) {
        HMODULE self_mod = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)&OrtGetApiBase, &self_mod);
        wchar_t self[MAX_PATH]; GetModuleFileNameW(self_mod, self, MAX_PATH);
        wchar_t* slash = wcsrchr(self, L'\\');
        if (slash) *(slash + 1) = 0;
        wchar_t real_path[MAX_PATH];
        swprintf_s(real_path, L"%lsonnxruntime_real.dll", self);

        g_real_module = LoadLibraryW(real_path);
        if (!g_real_module) {
            log_line("[fatal] LoadLibraryW failed err=%lu", GetLastError());
            return nullptr;
        }
        OrtGetApiBaseFn get_base = (OrtGetApiBaseFn)GetProcAddress(g_real_module, "OrtGetApiBase");
        if (!get_base) {
            log_line("[fatal] OrtGetApiBase not found in real runtime");
            return nullptr;
        }
        g_real_base = (OrtApiBase*)get_base();
        const OrtApi* real = g_real_base->GetApi(ORT_API_VERSION);
        if (!real) {
            log_line("[fatal] real GetApi(%d) returned null", ORT_API_VERSION);
            return g_real_base;
        }
        memcpy(&g_api, real, sizeof(OrtApi));
        g_orig_csfa    = g_api.CreateSessionFromArray;
        g_orig_csfa_pw = g_api.CreateSessionFromArrayWithPrepackedWeightsContainer;
        g_orig_ctwda = g_api.CreateTensorWithDataAsOrtValue;
    g_api.CreateTensorWithDataAsOrtValue = Hook_CreateTensorWithDataAsOrtValue;
    g_orig_run = g_api.Run;
        g_api.Run = Hook_Run;
        InitializeCriticalSection(&g_io_lock);
        g_api.CreateSessionFromArray = Hook_CreateSessionFromArray;
        g_api.CreateSessionFromArrayWithPrepackedWeightsContainer =
            Hook_CreateSessionFromArrayWithPrepackedWeightsContainer;

        g_fake_base.GetVersionString = Hook_GetVersionString;
        g_fake_base.GetApi           = Hook_GetApi;

        g_ready = true;
        char ver[64] = {0};
        if (g_real_base->GetVersionString) {
            strncpy_s(ver, g_real_base->GetVersionString(), _TRUNCATE);
        }
        log_line("[init] real ORT version=%s api_version=%d sizeof(OrtApi)=%zu",
                 ver, ORT_API_VERSION, sizeof(OrtApi));
    }
    return &g_fake_base;
}


BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) ensure_lock();
    return TRUE;
}














