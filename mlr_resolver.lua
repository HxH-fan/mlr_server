-- Пути, где может находиться основной файл скрипта
local candidate_paths = {
    "lua/resolver.lua",
    "resolver.lua",
    -- Добавьте сюда другие возможные пути, если нужно
}

local script_content = nil
local loaded_path = nil

-- Пробуем прочитать файл по каждому из путей
for _, path in ipairs(candidate_paths) do
    -- Используем pcall, чтобы перехватить ошибку, если файл не найден
    local success, content = pcall(readfile, path)
    if success and content then
        script_content = content
        loaded_path = path
        break
    end
end

-- Если файл не найден, выводим ошибку
if not script_content then
    error("Не удалось найти файл resolver.lua. Проверьте пути:\n" .. table.concat(candidate_paths, "\n"))
end

-- Компилируем и выполняем загруженный код
local chunk, load_err = load(script_content, loaded_path)
if not chunk then
    error("Ошибка компиляции скрипта " .. loaded_path .. ": " .. load_err)
end

-- Выполняем скрипт
chunk()