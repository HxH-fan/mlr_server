client.log("=== resolver.lua loaded ===")
local bit = require("bit")
local json = require("json")
local ffi = require("ffi")

-- ДИАГНОСТИКА: проверяем загрузку WinHTTP
client.log("[DIAG] ffi loaded:", ffi ~= nil)
local HAS_WINHTTP, WINHTTP = false, nil
-- Пробуем полный путь к winhttp.dll (64-битная система)
pcall(function()
    WINHTTP = ffi.load("C:\\Windows\\System32\\winhttp.dll")
    HAS_WINHTTP = WINHTTP ~= nil
end)
-- Если не вышло, пробуем путь для 32-бит на 64-бит
if not HAS_WINHTTP then
    pcall(function()
        WINHTTP = ffi.load("C:\\Windows\\SysWOW64\\winhttp.dll")
        HAS_WINHTTP = WINHTTP ~= nil
    end)
end
-- Если всё равно нет, пробуем стандартное имя (вдруг сработает)
if not HAS_WINHTTP then
    pcall(function()
        WINHTTP = ffi.load("winhttp")
        HAS_WINHTTP = WINHTTP ~= nil
    end)
end
client.log("[DIAG] HAS_WINHTTP after full path =", HAS_WINHTTP)
client.log("[DIAG] HAS_WINHTTP =", HAS_WINHTTP)
if not HAS_WINHTTP then
    client.log("[DIAG] WinHTTP load error:", WINHTTP)
end

local HAS_GS_HTTP, gs_http = pcall(require, "gamesense/http")

pcall(ffi.cdef, [[
typedef void* HINTERNET;
typedef const wchar_t* LPCWSTR;
typedef wchar_t* LPWSTR;
typedef unsigned short INTERNET_PORT;
typedef unsigned long DWORD;
typedef unsigned long DWORD_PTR;
typedef int BOOL;
typedef const void* LPCVOID;
typedef void* LPVOID;

HINTERNET WinHttpOpen(LPCWSTR pszAgentW, DWORD dwAccessType, LPCWSTR pszProxyW, LPCWSTR pszProxyBypassW, DWORD dwFlags);
BOOL WinHttpCloseHandle(HINTERNET hInternet);
BOOL WinHttpSetTimeouts(HINTERNET hInternet, int nResolveTimeout, int nConnectTimeout, int nSendTimeout, int nReceiveTimeout);
HINTERNET WinHttpConnect(HINTERNET hSession, LPCWSTR pswzServerName, INTERNET_PORT nServerPort, DWORD dwReserved);
HINTERNET WinHttpOpenRequest(HINTERNET hConnect, LPCWSTR pwszVerb, LPCWSTR pwszObjectName, LPCWSTR pwszVersion, LPCWSTR pwszReferrer, LPCWSTR *ppwszAcceptTypes, DWORD dwFlags);
BOOL WinHttpSendRequest(HINTERNET hRequest, LPCWSTR lpszHeaders, DWORD dwHeadersLength, LPVOID lpOptional, DWORD dwOptionalLength, DWORD dwTotalLength, DWORD_PTR dwContext);
BOOL WinHttpReceiveResponse(HINTERNET hRequest, LPVOID lpReserved);
BOOL WinHttpQueryHeaders(HINTERNET hRequest, DWORD dwInfoLevel, LPCWSTR pwszName, LPVOID lpBuffer, DWORD* lpdwBufferLength, DWORD* lpdwIndex);
BOOL WinHttpQueryDataAvailable(HINTERNET hRequest, DWORD* lpdwNumberOfBytesAvailable);
BOOL WinHttpReadData(HINTERNET hRequest, LPVOID lpBuffer, DWORD dwNumberOfBytesToRead, DWORD* lpdwNumberOfBytesRead);
DWORD GetLastError(void);
]])
local WINHTTP_ACCESS_TYPE_NO_PROXY = 1
local WINHTTP_FLAG_SECURE = 0x00800000
local WINHTTP_QUERY_STATUS_CODE = 19
local WINHTTP_QUERY_FLAG_NUMBER = 0x20000000

local URL = "http://127.0.0.1:8080"
local PREDICT_URL = URL .. "/predict"
local FEEDBACK_URL = URL .. "/feedback"

local THREAT_REQUEST_INTERVAL = 0.035
local THREAT_ACTIVE_REQUEST_INTERVAL = 0.022
local BACKGROUND_REQUEST_INTERVAL = 0.11
local BACKGROUND_ACTIVE_REQUEST_INTERVAL = 0.07
local REQUEST_TIMEOUT = 0.12
local HARD_REQUEST_TIMEOUT = 0.30
local SERVER_RESULT_TTL = 0.20
local CACHE_TTL = 0.80
local HTTP_TIMEOUT_MS = 150
local MAX_HTTP_QUEUE = 32
local HISTORY_SIZE = 8
local CACHE_SIZE = 12
local ENEMY_LIST_TTL = 0.10
local THREAT_SAMPLE_INTERVAL = 0.02
local THREAT_ACTIVE_SAMPLE_INTERVAL = 0.012
local BACKGROUND_SAMPLE_INTERVAL = 0.08
local BACKGROUND_ACTIVE_SAMPLE_INTERVAL = 0.045
local BACKGROUND_RESOLVE_PER_TICK = 1
local BACKGROUND_RESOLVE_PER_TICK_MAX = 2
local HTTP_PUMP_BUDGET = 3
local RESEND_YAW_DELTA = 6
local RESEND_SPEED_DELTA = 14
local RESEND_DISTANCE_DELTA = 48
local RESEND_LAST_DELTA = 6
local FAST_STATE_SPEED = 95
local FAST_STATE_YAW_DELTA = 14

local STATE_VECTOR_SIZE = 18
local DERIVED_FEATURE_SIZE = 12
local LOCAL_ACTIONS = {-120, -90, -60, -30, 0, 30, 60, 90, 120}
local BRUTE_ACTIONS = {0, 30, -30, 60, -60, 90, -90, 120, -120}

local WALL_TIME_BASE = (client.unix_time and client.unix_time() or client.system_time()) - globals.realtime()

local players = {}
local shots = {}
local shot_uid = 0
local http_queue = {
    items = {},
    head = 1,
    tail = 0,
    size = 0,
}
local enemy_cache = {
    list = {},
    updated_at = -1,
    cursor = 0,
}
local http_transport_logged = false
client.log("[DIAG] HAS_GS_HTTP =", HAS_GS_HTTP)
local function now()
    return globals.realtime()
end

local function unix_time()
    return WALL_TIME_BASE + now()
end

local function clamp(value, low, high)
    if value < low then
        return low
    end
    if value > high then
        return high
    end
    return value
end

local function norm(angle)
    while angle > 180 do
        angle = angle - 360
    end
    while angle < -180 do
        angle = angle + 360
    end
    return angle
end

local function angle_delta(a, b)
    return norm((a or 0) - (b or 0))
end

local function angle_distance(a, b)
    return math.abs(angle_delta(a, b))
end

local function round4(value)
    return math.floor((value or 0) * 10000 + 0.5) / 10000
end

local function log_http(...)
    client.log("[resolver/http] ", ...)
end

local function push_recent(list, value, limit)
    list[#list + 1] = value
    while #list > limit do
        table.remove(list, 1)
    end
end

local function to_wide(value)
    local text = tostring(value or "")
    local buffer = ffi.new("wchar_t[?]", #text + 1)
    for i = 1, #text do
        buffer[i - 1] = text:byte(i)
    end
    buffer[#text] = 0
    return buffer, #text
end

local function parse_url(url)
    local scheme, rest = tostring(url or ""):match("^(https?)://(.+)$")
    if not scheme then
        return nil
    end

    local host_port, path = rest:match("^([^/]+)(/.*)$")
    if not host_port then
        host_port = rest
        path = "/"
    end

    local host, port = host_port:match("^([^:]+):(%d+)$")
    if not host then
        host = host_port
        port = scheme == "https" and 443 or 80
    end

    return {
        scheme = scheme,
        host = host,
        port = tonumber(port),
        path = path,
    }
end

local function is_loopback_url(url)
    return url:match("^https?://127%.0%.0%.1[:/]") ~= nil
        or url:match("^https?://localhost[:/]") ~= nil
end

local function build_header_string(headers)
    local lines = {}
    for key, value in pairs(headers or {}) do
        lines[#lines + 1] = tostring(key) .. ": " .. tostring(value)
    end
    if #lines == 0 then
        return nil
    end
    return table.concat(lines, "\r\n")
end

local function winhttp_status_code(request)
    local status = ffi.new("DWORD[1]", 0)
    local size = ffi.new("DWORD[1]", ffi.sizeof(status))
    local ok = WINHTTP.WinHttpQueryHeaders(
        request,
        bit.bor(WINHTTP_QUERY_STATUS_CODE, WINHTTP_QUERY_FLAG_NUMBER),
        nil,
        status,
        size,
        nil
    )

    if ok == 0 then
        return 0
    end
    return tonumber(status[0]) or 0
end

local function winhttp_read_body(request)
    local chunks = {}
    local available = ffi.new("DWORD[1]", 0)

    while WINHTTP.WinHttpQueryDataAvailable(request, available) ~= 0 do
        local size = tonumber(available[0]) or 0
        if size <= 0 then
            break
        end

        local buffer = ffi.new("char[?]", size)
        local read = ffi.new("DWORD[1]", 0)
        if WINHTTP.WinHttpReadData(request, buffer, size, read) == 0 then
            break
        end

        local read_size = tonumber(read[0]) or 0
        if read_size <= 0 then
            break
        end
        chunks[#chunks + 1] = ffi.string(buffer, read_size)
    end

    return table.concat(chunks)
end

local function winhttp_post(url, body, headers)
    if not HAS_WINHTTP then
        return false, {status = 0, body = "", error = "winhttp_unavailable"}
    end

    local parsed = parse_url(url)
    if not parsed then
        return false, {status = 0, body = "", error = "invalid_url"}
    end

    local session = nil
    local connection = nil
    local request = nil
    local function has_handle(handle)
        return handle ~= nil and handle ~= ffi.NULL
    end

    local function close_handles()
        if has_handle(request) then
            WINHTTP.WinHttpCloseHandle(request)
            request = nil
        end
        if has_handle(connection) then
            WINHTTP.WinHttpCloseHandle(connection)
            connection = nil
        end
        if has_handle(session) then
            WINHTTP.WinHttpCloseHandle(session)
            session = nil
        end
    end

    local agent = to_wide("MLRResolver/1.0")
    session = WINHTTP.WinHttpOpen(agent, WINHTTP_ACCESS_TYPE_NO_PROXY, nil, nil, 0)
    if not has_handle(session) then
        return false, {status = 0, body = "", error = "session_open_failed"}
    end
    WINHTTP.WinHttpSetTimeouts(session, HTTP_TIMEOUT_MS, HTTP_TIMEOUT_MS, HTTP_TIMEOUT_MS, HTTP_TIMEOUT_MS)

    local host = to_wide(parsed.host)
    connection = WINHTTP.WinHttpConnect(session, host, parsed.port, 0)
    if not has_handle(connection) then
        local response = {status = 0, body = "", error = "connect_failed", winerror = tonumber(WINHTTP.GetLastError()) or 0}
        close_handles()
        return false, response
    end

    local method = to_wide("POST")
    local object_name = to_wide(parsed.path)
    local flags = parsed.scheme == "https" and WINHTTP_FLAG_SECURE or 0
    request = WINHTTP.WinHttpOpenRequest(connection, method, object_name, nil, nil, nil, flags)
    if not has_handle(request) then
        local response = {status = 0, body = "", error = "request_open_failed", winerror = tonumber(WINHTTP.GetLastError()) or 0}
        close_handles()
        return false, response
    end
    WINHTTP.WinHttpSetTimeouts(request, HTTP_TIMEOUT_MS, HTTP_TIMEOUT_MS, HTTP_TIMEOUT_MS, HTTP_TIMEOUT_MS)

    local header_text = build_header_string(headers)
    local header_wide, header_len = nil, 0
    if header_text ~= nil then
        header_wide, header_len = to_wide(header_text)
    end

    local payload = tostring(body or "")
    local payload_len = #payload
    local payload_buffer = nil
    if payload_len > 0 then
        payload_buffer = ffi.new("uint8_t[?]", payload_len)
        ffi.copy(payload_buffer, payload, payload_len)
    end

    local sent = WINHTTP.WinHttpSendRequest(
        request,
        header_wide,
        header_len,
        payload_buffer,
        payload_len,
        payload_len,
        0
    )
    if sent == 0 then
        local response = {status = 0, body = "", error = "send_failed", winerror = tonumber(WINHTTP.GetLastError()) or 0}
        close_handles()
        return false, response
    end

    if WINHTTP.WinHttpReceiveResponse(request, nil) == 0 then
        local response = {status = 0, body = "", error = "receive_failed", winerror = tonumber(WINHTTP.GetLastError()) or 0}
        close_handles()
        return false, response
    end

    local response = {
        status = winhttp_status_code(request),
        body = winhttp_read_body(request),
    }
    close_handles()
    return response.status ~= 0, response
end

local function http_backend_for_url(url)
    if is_loopback_url(url) and HAS_WINHTTP then
        return "winhttp"
    end
    if HAS_GS_HTTP then
        return "gamesense"
    end
    return HAS_WINHTTP and "winhttp" or "none"
end

local dequeue_http_post
local function enqueue_http_post(url, body, headers, callback)
    if http_queue.size >= MAX_HTTP_QUEUE then
        if url ~= FEEDBACK_URL then
            return false
        end
        dequeue_http_post()
    end

    http_queue.tail = http_queue.tail + 1
    http_queue.items[http_queue.tail] = {
        url = url,
        body = body or "",
        headers = headers or {},
        callback = callback,
        backend = http_backend_for_url(url),
        kind = url == FEEDBACK_URL and "feedback" or "predict",
    }
    http_queue.size = http_queue.size + 1
    return true
end

dequeue_http_post = function()
    if http_queue.size <= 0 then
        return nil
    end

    local item = http_queue.items[http_queue.head]
    http_queue.items[http_queue.head] = nil
    http_queue.head = http_queue.head + 1
    http_queue.size = http_queue.size - 1
    if http_queue.size <= 0 then
        http_queue.head = 1
        http_queue.tail = 0
        http_queue.size = 0
    end
    return item
end

local function process_http_queue(limit)
    local budget = limit or 1
    while budget > 0 do
        local item = dequeue_http_post()
        if item == nil then
            return
        end
        if not http_transport_logged then
            http_transport_logged = true
            log_http("using backend=", item.backend, " for ", item.url)
        end
        if item.backend == "gamesense" and HAS_GS_HTTP then
            gs_http.post(item.url, {
                headers = item.headers,
                body = item.body,
            }, function(success, response)
                if item.callback ~= nil then
                    item.callback(success, response)
                end
            end)
        elseif item.backend == "none" then
            if item.callback ~= nil then
                item.callback(false, {status = 0, body = "", error = "no_http_backend"})
            end
        else
            local success, response = winhttp_post(item.url, item.body, item.headers)
            if item.callback ~= nil then
                item.callback(success, response)
            end
        end
        budget = budget - 1
    end
end

local function fnv1a32(text)
    local hash = 2166136261
    for i = 1, #text do
        hash = bit.bxor(hash, text:byte(i))
        hash = (hash * 16777619) % 4294967296
    end
    return math.floor(hash)
end

local function snap_action(action)
    local best = LOCAL_ACTIONS[1]
    local best_diff = angle_distance(action, best)
    for i = 2, #LOCAL_ACTIONS do
        local candidate = LOCAL_ACTIONS[i]
        local diff = angle_distance(action, candidate)
        if diff < best_diff then
            best = candidate
            best_diff = diff
        end
    end
    return best
end

local function distance3(ax, ay, az, bx, by, bz)
    local dx = (bx or 0) - (ax or 0)
    local dy = (by or 0) - (ay or 0)
    local dz = (bz or 0) - (az or 0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)
end

local function calculate_view_angles(from_x, from_y, from_z, to_x, to_y, to_z)
    local dx = (to_x or 0) - (from_x or 0)
    local dy = (to_y or 0) - (from_y or 0)
    local dz = (to_z or 0) - (from_z or 0)
    local hyp = math.sqrt(dx * dx + dy * dy)
    local pitch = -math.deg(math.atan2(dz, hyp))
    local yaw = math.deg(math.atan2(dy, dx))
    return pitch, yaw
end

local function new_request_state()
    return {
        in_flight = false,
        sent_at = 0,
        shot_id = 0,
    }
end

local function get_player_key(p)
    local steam64 = entity.get_steam64(p)
    if steam64 and steam64 ~= 0 then
        return tostring(steam64)
    end
    return "ent:" .. tostring(p)
end

local function get_player_id(p)
    return fnv1a32(get_player_key(p))
end

local function get_eye_yaw(p)
    local _, yaw = entity.get_prop(p, "m_angEyeAngles")
    return norm(tonumber(yaw) or 0)
end

local function get_motion_snapshot(p)
    local vx = tonumber(entity.get_prop(p, "m_vecVelocity[0]")) or 0
    local vy = tonumber(entity.get_prop(p, "m_vecVelocity[1]")) or 0
    local vz = tonumber(entity.get_prop(p, "m_vecVelocity[2]")) or 0
    local speed_2d = math.sqrt(vx * vx + vy * vy)
    local duck_amount = tonumber(entity.get_prop(p, "m_flDuckAmount")) or 0
    local flags = entity.get_prop(p, "m_fFlags") or 0
    local on_ground = bit.band(flags, 1) == 1

    return {
        vx = vx,
        vy = vy,
        vz = vz,
        speed_2d = speed_2d,
        duck_amount = clamp(duck_amount, 0, 1),
        on_ground = on_ground,
    }
end

local function classify_state(motion)
    if not motion.on_ground then
        return "AIR"
    end
    if motion.speed_2d > 5 then
        return "MOVING"
    end
    return "STANDING"
end

local function get_target_position(p)
    if type(entity.hitbox_position) == "function" then
        local x, y, z = entity.hitbox_position(p, 0)
        if x ~= nil then
            return x, y, z
        end
    end

    if type(entity.get_origin) == "function" then
        local x, y, z = entity.get_origin(p)
        if x ~= nil then
            return x, y, z
        end
    end

    return nil
end

local function refresh_enemy_cache(current_time)
    if current_time - enemy_cache.updated_at < ENEMY_LIST_TTL then
        return enemy_cache.list
    end

    local raw = entity.get_players(true)
    local filtered = {}
    for i = 1, #raw do
        local enemy = raw[i]
        if entity.is_alive(enemy) then
            filtered[#filtered + 1] = enemy
        end
    end

    enemy_cache.list = filtered
    enemy_cache.updated_at = current_time
    if enemy_cache.cursor > #filtered then
        enemy_cache.cursor = 0
    end
    return filtered
end

local function get_current_threat(enemies)
    if type(client.current_threat) == "function" then
        return client.current_threat()
    end

    if type(client.eye_position) ~= "function"
        or type(client.camera_angles) ~= "function"
        or type(entity.hitbox_position) ~= "function" then
        return nil
    end

    local eye_x, eye_y, eye_z = client.eye_position()
    local view_pitch, view_yaw = client.camera_angles()
    if eye_x == nil or view_pitch == nil then
        return nil
    end

    local best_target = nil
    local best_score = 180
    local pool = enemies or entity.get_players(true)
    for i = 1, #pool do
        local enemy = pool[i]
        local hitbox_x, hitbox_y, hitbox_z = entity.hitbox_position(enemy, 0)
        if hitbox_x ~= nil then
            local pitch, yaw = calculate_view_angles(eye_x, eye_y, eye_z, hitbox_x, hitbox_y, hitbox_z)
            local score = angle_distance(yaw, view_yaw) + angle_distance(pitch, view_pitch) * 0.5
            if score < best_score then
                best_score = score
                best_target = enemy
            end
        end
    end
    return best_target
end

local function next_background_enemy(enemies, threat)
    local count = #enemies
    if count <= 0 then
        return nil
    end

    for _ = 1, count do
        enemy_cache.cursor = (enemy_cache.cursor % count) + 1
        local enemy = enemies[enemy_cache.cursor]
        if enemy ~= nil and enemy ~= threat then
            return enemy
        end
    end
    return nil
end

local function new_player_record(p)
    return {
        player_id = get_player_id(p),
        player_key = get_player_key(p),
        current_shot_id = 1,
        yaw_history = {},
        feature_history = {},
        cache = {},
        request = new_request_state(),
        server_result = nil,
        state = "STANDING",
        air_time = 0,
        duck_amount = 0,
        speed_2d = 0,
        speed_z = 0,
        relative_yaw = 0,
        aim_pitch = 0,
        target_distance = 0,
        last = 0,
        last_delta = 0,
        avg_delta = 0,
        jitter_score = 0,
        variance_score = 0,
        current_state_vector = {},
        current_derived = {},
        last_action = 0,
        last_success_action = nil,
        last_confidence = 0,
        last_sample_at = -1,
        last_sent_at = -1,
        last_sent_shot_id = 0,
        last_sent_state = nil,
        last_sent_relative_yaw = 0,
        last_sent_speed_2d = 0,
        last_sent_distance = 0,
        last_sent_delta = 0,
        mode = "FS",
        misses = 0,
        brute_index = 1,
        cache_used = false,
        fallback_used = false,
    }
end

local function get(p)
    local key = get_player_key(p)
    if not players[key] then
        players[key] = new_player_record(p)
    end
    return players[key]
end

local function recent_delta_stats(hist)
    if #hist < 2 then
        return 0, 0, 0
    end

    local deltas = {}
    local total = 0
    local max_delta = 0
    for i = 2, #hist do
        local delta = angle_delta(hist[i], hist[i - 1])
        deltas[#deltas + 1] = delta
        total = total + delta
        local abs_delta = math.abs(delta)
        if abs_delta > max_delta then
            max_delta = abs_delta
        end
    end

    local average = total / #deltas
    local variance = 0
    for i = 1, #deltas do
        local diff = deltas[i] - average
        variance = variance + diff * diff
    end
    variance = variance / #deltas
    return deltas[#deltas] or 0, average, max_delta, variance
end

local function build_state_vector(rec)
    return {
        round4(clamp((rec.target_dx or 0) / 4096, -1.5, 1.5)),
        round4(clamp((rec.target_dy or 0) / 4096, -1.5, 1.5)),
        round4(clamp((rec.target_dz or 0) / 1024, -1.5, 1.5)),
        round4(clamp((rec.target_vx or 0) / 400, -1.5, 1.5)),
        round4(clamp((rec.target_vy or 0) / 400, -1.5, 1.5)),
        round4(clamp((rec.target_vz or 0) / 400, -1.5, 1.5)),
        round4(clamp((rec.last or 0) / 180, -1, 1)),
        round4(clamp((rec.last_delta or 0) / 180, -1, 1)),
        round4(clamp((rec.avg_delta or 0) / 180, -1, 1)),
        round4(clamp((rec.speed_2d or 0) / 320, 0, 1.5)),
        round4(clamp(rec.duck_amount or 0, 0, 1)),
        round4(clamp((rec.air_time or 0) / 1.5, 0, 1.5)),
        rec.state == "STANDING" and 1 or 0,
        rec.state == "MOVING" and 1 or 0,
        rec.state == "AIR" and 1 or 0,
        round4(clamp((rec.relative_yaw or 0) / 180, -1, 1)),
        round4(clamp((rec.aim_pitch or 0) / 90, -1, 1)),
        round4(clamp((rec.target_distance or 0) / 3000, 0, 2)),
    }
end

local function build_derived_features(rec)
    return {
        round4(rec.target_distance or 0),
        round4(rec.relative_yaw or 0),
        round4(rec.last_delta or 0),
        rec.state == "STANDING" and 1 or 0,
        rec.state == "MOVING" and 1 or 0,
        rec.state == "AIR" and 1 or 0,
        round4(rec.speed_2d or 0),
        round4(rec.speed_z or 0),
        round4(rec.avg_delta or 0),
        round4(rec.jitter_score or 0),
        round4(rec.variance_score or 0),
        round4(math.abs(rec.aim_pitch or 0)),
    }
end

local function history_payload(rec)
    local output = {}
    local start = math.max(1, #rec.feature_history - HISTORY_SIZE + 1)
    for i = start, #rec.feature_history do
        output[#output + 1] = rec.feature_history[i]
    end
    return output
end

local function record_player(p, current_time, eye_x, eye_y, eye_z, sample_interval)
    local rec = get(p)
    if rec.last_sample_at >= 0
        and current_time - rec.last_sample_at < (sample_interval or BACKGROUND_SAMPLE_INTERVAL)
        and rec.current_state_vector[1] ~= nil then
        return rec, false
    end

    local motion = get_motion_snapshot(p)
    local target_x, target_y, target_z = get_target_position(p)
    if eye_x == nil or target_x == nil then
        return rec, false
    end

    rec.last = get_eye_yaw(p)
    rec.state = classify_state(motion)
    rec.duck_amount = motion.duck_amount or 0
    rec.speed_2d = motion.speed_2d or 0
    rec.speed_z = motion.vz or 0
    rec.target_vx = motion.vx or 0
    rec.target_vy = motion.vy or 0
    rec.target_vz = motion.vz or 0

    if rec.state == "AIR" then
        rec.air_time = clamp((rec.air_time or 0) + globals.tickinterval(), 0, 1.5)
    else
        rec.air_time = 0
    end

    rec.target_dx = target_x - eye_x
    rec.target_dy = target_y - eye_y
    rec.target_dz = target_z - eye_z
    rec.target_distance = distance3(eye_x, eye_y, eye_z, target_x, target_y, target_z)

    local aim_pitch, yaw_to_target = calculate_view_angles(eye_x, eye_y, eye_z, target_x, target_y, target_z)
    local _, yaw_to_local = calculate_view_angles(target_x, target_y, target_z, eye_x, eye_y, eye_z)
    rec.relative_yaw = angle_delta(rec.last, yaw_to_local)
    rec.aim_pitch = aim_pitch or 0

    push_recent(rec.yaw_history, rec.last, HISTORY_SIZE + 4)
    rec.last_delta, rec.avg_delta, rec.jitter_score, rec.variance_score = recent_delta_stats(rec.yaw_history)
    rec.current_state_vector = build_state_vector(rec)
    rec.current_derived = build_derived_features(rec)
    push_recent(rec.feature_history, rec.current_state_vector, HISTORY_SIZE + 2)
    rec.last_sample_at = current_time
    return rec, true
end

local function push_cache(rec, entry)
    rec.cache[#rec.cache + 1] = entry
    while #rec.cache > CACHE_SIZE do
        table.remove(rec.cache, 1)
    end
end

local function find_cache_hit(rec, current_time)
    local best = nil
    local best_score = -999
    for i = #rec.cache, 1, -1 do
        local entry = rec.cache[i]
        local age = current_time - (entry.created_at or 0)
        if age <= CACHE_TTL and entry.state == rec.state then
            local score = (entry.confidence or 0) - age * 0.4
            if entry.hit == true then
                score = score + 0.20
            elseif entry.hit == false then
                score = score - 0.10
            end
            if score > best_score then
                best_score = score
                best = entry
            end
        end
    end
    return best
end

local function can_use_server_result(rec, current_time)
    local result = rec.server_result
    if not result then
        return false
    end
    if result.shot_id ~= rec.current_shot_id then
        return false
    end
    if current_time - (result.received_at or 0) > SERVER_RESULT_TTL then
        return false
    end
    return result.state == rec.state
end

local function build_predict_payload(rec)
    return {
        player_id = rec.player_id,
        shot_id = rec.current_shot_id,
        timestamp = round4(unix_time()),
        state_vector = rec.current_state_vector,
        derived_features = rec.current_derived,
        history = history_payload(rec),
        player_state = rec.state,
    }
end

local should_fetch_server
local function fetch_server(rec, current_time, request_interval)
    current_time = current_time or now()
    request_interval = request_interval or BACKGROUND_REQUEST_INTERVAL
    if rec.request.in_flight and current_time - rec.request.sent_at > HARD_REQUEST_TIMEOUT then
        rec.request.in_flight = false
    end

    if rec.request.in_flight or current_time - (rec.request.sent_at or 0) < request_interval then
        return
    end

    if not should_fetch_server(rec, current_time) then
        return
    end

    local payload = build_predict_payload(rec)
    rec.request.in_flight = true
    rec.request.sent_at = current_time
    rec.request.shot_id = payload.shot_id
    rec.last_sent_at = current_time
    rec.last_sent_shot_id = payload.shot_id
    rec.last_sent_state = rec.state
    rec.last_sent_relative_yaw = rec.relative_yaw or 0
    rec.last_sent_speed_2d = rec.speed_2d or 0
    rec.last_sent_distance = rec.target_distance or 0
    rec.last_sent_delta = rec.last_delta or 0

    local queued = enqueue_http_post(PREDICT_URL, json.stringify(payload), {
        ["Content-Type"] = "application/json",
        ["Connection"] = "close",
    }, function(success, response)
        rec.request.in_flight = false
        if not success or not response or response.status ~= 200 then
            if response and response.error then
                log_http("predict failed status=", response.status or 0, " error=", response.error)
            end
            return
        end

        local ok, data = pcall(json.parse, response.body)
        if not ok or not data then
            return
        end

        local predicted_action = tonumber(data.predicted_action)
        local confidence = tonumber(data.confidence)
        local strategy = data.strategy_used
        local shot_id = tonumber(data.shot_id) or payload.shot_id
        if predicted_action == nil or confidence == nil or strategy == nil then
            return
        end

        local result = {
            shot_id = shot_id,
            action = snap_action(predicted_action),
            confidence = clamp(confidence, 0, 1),
            strategy = strategy,
            state = payload.player_state,
            received_at = now(),
            created_at = now(),
        }
        push_cache(rec, result)
        if shot_id == rec.current_shot_id then
            rec.server_result = result
        end
    end)
    if not queued then
        rec.request.in_flight = false
    end
end

local function choose_local_fallback(rec)
    local fs_action = snap_action(clamp(-(rec.relative_yaw or 0) * 0.65, -120, 120))
    local last_action = snap_action(rec.last_success_action or rec.last_action or 0)
    local brute_action = BRUTE_ACTIONS[rec.brute_index] or 0

    local best = {
        strategy = "FS",
        action = fs_action,
        confidence = clamp(0.25 + (1 - clamp(math.abs(rec.last_delta or 0) / 180, 0, 1)) * 0.30, 0.18, 0.74),
    }

    if rec.last_success_action ~= nil then
        local last_conf = clamp(0.35 + (rec.misses == 0 and 0.15 or 0), 0.22, 0.78)
        if last_conf > best.confidence then
            best = {strategy = "LAST", action = last_action, confidence = last_conf}
        end
    end

    local brute_conf = clamp(0.22 + (rec.misses >= 2 and 0.26 or 0.08), 0.16, 0.72)
    if brute_conf > best.confidence then
        best = {strategy = "BRUTE", action = brute_action, confidence = brute_conf}
    end

    return best
end

local function choose_resolution(rec, current_time)
    current_time = current_time or now()
    local action = nil
    local strategy = nil
    local confidence = 0
    local used_cache = false
    local used_fallback = false

    if rec.request.in_flight and current_time - rec.request.sent_at > REQUEST_TIMEOUT then
        rec.request.in_flight = false
    end

    if can_use_server_result(rec, current_time) then
        action = rec.server_result.action
        strategy = rec.server_result.strategy
        confidence = rec.server_result.confidence
    else
        local cache_hit = find_cache_hit(rec, current_time)
        if cache_hit then
            action = cache_hit.action
            strategy = cache_hit.strategy
            confidence = clamp((cache_hit.confidence or 0.4) * 0.92, 0, 1)
            used_cache = true
        else
            local local_choice = choose_local_fallback(rec)
            action = local_choice.action
            strategy = local_choice.strategy
            confidence = local_choice.confidence
            used_fallback = true
        end
    end

    rec.last_action = snap_action(action or 0)
    rec.mode = strategy or "FS"
    rec.last_confidence = confidence or 0
    rec.cache_used = used_cache
    rec.fallback_used = used_fallback
end

local function has_meaningful_state_change(rec)
    if rec.last_sent_state == nil or rec.last_sent_state ~= rec.state then
        return true
    end
    if math.abs((rec.relative_yaw or 0) - (rec.last_sent_relative_yaw or 0)) >= RESEND_YAW_DELTA then
        return true
    end
    if math.abs((rec.speed_2d or 0) - (rec.last_sent_speed_2d or 0)) >= RESEND_SPEED_DELTA then
        return true
    end
    if math.abs((rec.target_distance or 0) - (rec.last_sent_distance or 0)) >= RESEND_DISTANCE_DELTA then
        return true
    end
    if math.abs((rec.last_delta or 0) - (rec.last_sent_delta or 0)) >= RESEND_LAST_DELTA then
        return true
    end
    return false
end

local function is_fast_state(rec)
    if rec == nil then
        return false
    end
    if rec.state == "AIR" then
        return true
    end
    if (rec.speed_2d or 0) >= FAST_STATE_SPEED then
        return true
    end
    if math.abs(rec.last_delta or 0) >= FAST_STATE_YAW_DELTA then
        return true
    end
    return false
end

should_fetch_server = function(rec, current_time)
    if rec.last_sent_shot_id ~= rec.current_shot_id then
        return true
    end
    if rec.server_result == nil then
        return true
    end
    if current_time - (rec.server_result.received_at or 0) > SERVER_RESULT_TTL * 0.5 then
        return true
    end
    return has_meaningful_state_change(rec)
end

local function request_interval_for_priority(priority, rec)
    if priority == "threat" then
        if is_fast_state(rec) then
            return THREAT_ACTIVE_REQUEST_INTERVAL
        end
        return THREAT_REQUEST_INTERVAL
    end
    if is_fast_state(rec) then
        return BACKGROUND_ACTIVE_REQUEST_INTERVAL
    end
    return BACKGROUND_REQUEST_INTERVAL
end

local function sample_interval_for_priority(priority, rec)
    if priority == "threat" then
        if is_fast_state(rec) then
            return THREAT_ACTIVE_SAMPLE_INTERVAL
        end
        return THREAT_SAMPLE_INTERVAL
    end
    if is_fast_state(rec) then
        return BACKGROUND_ACTIVE_SAMPLE_INTERVAL
    end
    return BACKGROUND_SAMPLE_INTERVAL
end

local function resolve_player(p, priority, current_time, eye_x, eye_y, eye_z)
    local known = get(p)
    local rec = record_player(p, current_time, eye_x, eye_y, eye_z, sample_interval_for_priority(priority, known))
    if not rec then
        return
    end
    choose_resolution(rec, current_time)
    fetch_server(rec, current_time, request_interval_for_priority(priority, rec))
end

local function send_feedback(shot)
    enqueue_http_post(FEEDBACK_URL, json.stringify({
            player_id = shot.player_id,
            shot_id = shot.shot_id,
            hit = shot.hit and true or false,
            latency_ms = round4(shot.latency_ms or 0),
            strategy_used = shot.strategy,
            cache_used = shot.cache_used and true or false,
            fallback_used = shot.fallback_used and true or false,
        }), {
        ["Content-Type"] = "application/json",
        ["Connection"] = "close",
    }, function(success, response)
        if not success and response and response.error then
            log_http("feedback failed status=", response.status or 0, " error=", response.error)
        end
    end)
end

client.set_event_callback("weapon_fire", function(e)
    local me = entity.get_local_player()
    local attacker = client.userid_to_entindex(e.userid)
    if not attacker or attacker ~= me then
        return
    end

    local target = get_current_threat()
    if not target or target == 0 or not entity.is_enemy(target) then
        return
    end

    local rec = get(target)
    if not rec then
        return
    end

    shot_uid = shot_uid + 1
    shots[shot_uid] = {
        target = target,
        player_id = rec.player_id,
        shot_id = rec.current_shot_id,
        fired_at = now(),
        action = rec.last_action,
        strategy = rec.mode,
        confidence = rec.last_confidence or 0,
        state = rec.state,
        cache_used = rec.cache_used and true or false,
        fallback_used = rec.fallback_used and true or false,
        latency_ms = rec.request.sent_at > 0 and (now() - rec.request.sent_at) * 1000 or 0,
        hit = nil,
    }

    rec.current_shot_id = rec.current_shot_id + 1
    rec.server_result = nil
end)

client.set_event_callback("player_hurt", function(e)
    local me = entity.get_local_player()
    if client.userid_to_entindex(e.attacker) ~= me then
        return
    end

    local victim = client.userid_to_entindex(e.userid)
    for _, shot in pairs(shots) do
        if shot.target == victim and shot.hit == nil then
            shot.hit = true
            break
        end
    end
end)

local function process_shots(current_time)
    current_time = current_time or now()
    local to_remove = {}

    for id, shot in pairs(shots) do
        if shot.hit == nil and current_time - shot.fired_at > 1.0 then
            shot.hit = false
        end

        if shot.hit ~= nil then
            local rec = get(shot.target)
            if rec then
                if shot.hit then
                    rec.last_success_action = shot.action
                    rec.misses = 0
                    rec.brute_index = 1
                else
                    rec.misses = rec.misses + 1
                    if shot.strategy == "BRUTE" or rec.misses >= 2 then
                        rec.brute_index = (rec.brute_index % #BRUTE_ACTIONS) + 1
                    end
                end

                push_cache(rec, {
                    shot_id = shot.shot_id,
                    action = shot.action,
                    confidence = shot.confidence,
                    strategy = shot.strategy,
                    state = shot.state,
                    created_at = current_time,
                    hit = shot.hit and true or false,
                })
                send_feedback(shot)
            end
            to_remove[#to_remove + 1] = id
        end
    end

    for i = 1, #to_remove do
        shots[to_remove[i]] = nil
    end
end

local function background_resolve_budget(enemies, threat)
    local count = #enemies
    if count <= 1 then
        return 0
    end
    if http_queue.size <= 6 and count <= 4 then
        return BACKGROUND_RESOLVE_PER_TICK_MAX
    end
    if threat == nil and count <= 3 then
        return BACKGROUND_RESOLVE_PER_TICK_MAX
    end
    return BACKGROUND_RESOLVE_PER_TICK
end

local function resolver_update()
    local current_time = now()
    local enemies = refresh_enemy_cache(current_time)
    local threat = get_current_threat(enemies)
    local eye_x, eye_y, eye_z = client.eye_position()
    local resolved_background = 0
    local background_budget = background_resolve_budget(enemies, threat)

    if threat ~= nil and (not entity.is_enemy(threat) or not entity.is_alive(threat)) then
        threat = nil
    end

    if threat ~= nil then
        resolve_player(threat, "threat", current_time, eye_x, eye_y, eye_z)
    end

    while resolved_background < background_budget do
        local enemy = next_background_enemy(enemies, threat)
        if enemy == nil then
            break
        end
        resolve_player(enemy, "background", current_time, eye_x, eye_y, eye_z)
        resolved_background = resolved_background + 1
    end

    process_shots(current_time)
    process_http_queue(HTTP_PUMP_BUDGET)
end

client.set_event_callback("setup_command", resolver_update)
