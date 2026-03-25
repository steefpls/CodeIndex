"""Tests for the Lua chunker.

Covers all patterns: module tables, named functions, xLua hotfixes (single + batch),
variable-assigned functions, assignment methods, table field functions,
class-like OOP, EmmyLua annotations, constructor detection, class summaries.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indexer.chunker_lua import chunk_file_lua


def _make_large(lines: list[str], pad_to: int = 160) -> bytes:
    """Pad a list of source lines to exceed SMALL_FILE_LINE_THRESHOLD."""
    while len(lines) < pad_to:
        lines.append(f"-- padding line {len(lines)}")
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_small_file_single_chunk():
    """Small Lua files should produce a single whole_class chunk."""
    source = b"""\
local M = {}

function M.greet(name)
    print("Hello, " .. name)
end

return M
"""
    chunks = chunk_file_lua(source, "scripts/greeter.lua", "scripts")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "M"
    assert chunks[0].namespace == "scripts.greeter"
    print("[PASS] test_small_file_single_chunk")


def test_top_level_functions():
    """Top-level functions in a large file should produce function chunks."""
    lines = ["-- Large Lua file"]
    for i in range(20):
        lines.append(f"function func_{i}()")
        for j in range(8):
            lines.append(f"    local x_{j} = {j}")
        lines.append("end")
        lines.append("")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "scripts/bigfile.lua", "scripts")
    assert len(chunks) > 1
    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) == 20
    assert func_chunks[0].method_name == "func_0"
    print("[PASS] test_top_level_functions")


def test_module_methods():
    """Functions on a module table should be chunked as methods."""
    lines = ["local Utils = {}"]
    for i in range(20):
        lines.append(f"function Utils.method_{i}(self)")
        for j in range(8):
            lines.append(f"    local y_{j} = {j}")
        lines.append("end")
        lines.append("")
    lines.append("return Utils")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "scripts/utils.lua", "mod")
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) == 20
    assert all(c.class_name == "Utils" for c in method_chunks)
    assert method_chunks[0].method_name == "method_0"
    print("[PASS] test_module_methods")


def test_colon_methods():
    """Colon-style method declarations should be detected."""
    lines = ["local Player = {}"]
    for i in range(20):
        lines.append(f"function Player:action_{i}()")
        for j in range(8):
            lines.append(f"    self.val_{j} = {j}")
        lines.append("end")
        lines.append("")
    lines.append("return Player")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "game/player.lua", "game")
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) == 20
    assert all(c.class_name == "Player" for c in method_chunks)
    assert method_chunks[0].method_name == "action_0"
    print("[PASS] test_colon_methods")


def test_local_function():
    """local function declarations should be chunked."""
    lines = ["-- helpers"]
    for i in range(20):
        lines.append(f"local function helper_{i}(x)")
        for j in range(8):
            lines.append(f"    local r_{j} = x + {j}")
        lines.append("    return r_0")
        lines.append("end")
        lines.append("")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "lib/helpers.lua", "lib")
    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) == 20
    assert func_chunks[0].method_name == "helper_0"
    print("[PASS] test_local_function")


def test_empty_file_fallback():
    """An empty-ish file should still produce a chunk."""
    source = b"-- just a comment\n"
    chunks = chunk_file_lua(source, "scripts/empty.lua", "scripts")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    print("[PASS] test_empty_file_fallback")


def test_mixed_functions_and_methods():
    """Mix of top-level functions and module methods."""
    lines = ["local M = {}"]
    for i in range(10):
        lines.append(f"function M.mod_func_{i}()")
        for j in range(8):
            lines.append(f"    local v_{j} = {j}")
        lines.append("end")
        lines.append("")
    for i in range(10):
        lines.append(f"function top_func_{i}()")
        for j in range(8):
            lines.append(f"    local w_{j} = {j}")
        lines.append("end")
        lines.append("")
    lines.append("return M")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "scripts/mixed.lua", "scripts")
    methods = [c for c in chunks if c.chunk_type == "method"]
    functions = [c for c in chunks if c.chunk_type == "function"]
    assert len(methods) == 10
    assert len(functions) == 10
    assert all(c.class_name == "M" for c in methods)
    print("[PASS] test_mixed_functions_and_methods")


def test_xlua_hotfix_pattern():
    """Small xLua hotfix scripts should produce whole_class + hotfix method chunks."""
    source = b"""\
local util = require("xlua.util")

function hotfix_init()
    print("Hotfix loaded")
end

xlua.hotfix(CS.Game.PlayerController, "Update", function(self)
    self.speed = 10
end)
"""
    chunks = chunk_file_lua(source, "hotfix/player_fix.lua", "hotfix")
    # Should have whole_class for the surrounding code + hotfix method chunk
    assert any(c.chunk_type == "whole_class" for c in chunks)
    hotfix = [c for c in chunks if c.class_name == "PlayerController"]
    assert len(hotfix) == 1
    assert hotfix[0].method_name == "Update"
    assert hotfix[0].chunk_type == "method"
    print("[PASS] test_xlua_hotfix_pattern")


def test_namespace_derivation():
    """Namespace should be derived from file path."""
    source = b"local x = 1\n"
    chunks = chunk_file_lua(source, "game/ui/widgets/healthbar.lua", "game")
    assert chunks[0].namespace == "game.ui.widgets.healthbar"
    print("[PASS] test_namespace_derivation")


def test_preceding_comment_extraction():
    """Comments before functions should be captured as doc_comment."""
    lines = ["local M = {}"]
    for i in range(15):
        lines.append(f"-- Does thing {i}")
        lines.append(f"function M.thing_{i}()")
        for j in range(8):
            lines.append(f"    local z_{j} = {j}")
        lines.append("end")
        lines.append("")
    lines.append("return M")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "scripts/commented.lua", "scripts")
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) > 0
    assert method_chunks[0].doc_comment == "Does thing 0"
    print("[PASS] test_preceding_comment_extraction")


# ---------------------------------------------------------------------------
# xLua hotfix — single method form (large file)
# ---------------------------------------------------------------------------

def test_hotfix_single_method():
    """xlua.hotfix(CS.X.Y, 'Method', function() end) should create a method chunk."""
    lines = [
        "local util = require('xlua.util')",
        "",
        "xlua.hotfix(CS.Game.Combat.DamageSystem, 'CalculateDamage', function(self, attacker, defender)",
        "    local base = attacker.atk - defender.def",
        "    local crit = math.random() < 0.1",
        "    if crit then",
        "        base = base * 2",
        "    end",
        "    return math.max(0, base)",
        "end)",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "hotfix/combat_fix.lua", "hotfix")
    hotfix = [c for c in chunks if c.class_name == "DamageSystem"]
    assert len(hotfix) == 1
    assert hotfix[0].method_name == "CalculateDamage"
    assert hotfix[0].chunk_type == "method"
    assert hotfix[0].namespace == "Game.Combat"
    assert "xLua hotfix" in hotfix[0].doc_comment
    assert "DamageSystem" in hotfix[0].doc_comment
    print("[PASS] test_hotfix_single_method")


# ---------------------------------------------------------------------------
# xLua hotfix — batch table form
# ---------------------------------------------------------------------------

def test_hotfix_batch():
    """xlua.hotfix(CS.X.Y, { M1 = function() end, M2 = function() end }) should create per-method chunks."""
    lines = [
        "xlua.hotfix(CS.Game.Player, {",
        "    Update = function(self)",
        "        local pos = self:GetPosition()",
        "        local target = self:FindTarget()",
        "        if target then",
        "            self:MoveToward(target)",
        "        end",
        "    end,",
        "    OnDeath = function(self)",
        "        self:PlayAnimation('death')",
        "        self:DropLoot()",
        "        self:ScheduleRespawn()",
        "    end,",
        "})",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "hotfix/player_fix.lua", "hotfix")
    player_methods = [c for c in chunks if c.class_name == "Player" and c.chunk_type == "method"]
    assert len(player_methods) == 2
    names = {c.method_name for c in player_methods}
    assert names == {"Update", "OnDeath"}
    assert all("xLua hotfix" in c.doc_comment for c in player_methods)
    assert all(c.namespace == "Game" for c in player_methods)
    print("[PASS] test_hotfix_batch")


# ---------------------------------------------------------------------------
# Variable-assigned functions
# ---------------------------------------------------------------------------

def test_variable_assigned_function():
    """local f = function() end should produce a function chunk."""
    lines = [
        "local M = {}",
        "",
        "local calculate_damage = function(attacker, defender)",
        "    local base = attacker.atk - defender.def",
        "    local armor_reduction = defender.armor / (defender.armor + 100)",
        "    return math.max(0, base * (1 - armor_reduction))",
        "end",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "scripts/combat.lua", "scripts")
    func_chunks = [c for c in chunks if c.method_name == "calculate_damage"]
    assert len(func_chunks) == 1
    assert func_chunks[0].chunk_type == "function"
    print("[PASS] test_variable_assigned_function")


# ---------------------------------------------------------------------------
# Assignment-based methods (M.handler = function)
# ---------------------------------------------------------------------------

def test_assignment_method():
    """M.handler = function() end should produce a method chunk."""
    lines = [
        "local M = {}",
        "",
    ]
    for i in range(10):
        lines.append(f"M.handler_{i} = function(self, evt)")
        for j in range(8):
            lines.append(f"    local v_{j} = evt.data_{j}")
        lines.append("end")
        lines.append("")
    lines.append("return M")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "scripts/events.lua", "scripts")
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) == 10
    assert all(c.class_name == "M" for c in method_chunks)
    assert method_chunks[0].method_name == "handler_0"
    print("[PASS] test_assignment_method")


# ---------------------------------------------------------------------------
# CS.X.Y = function() direct override
# ---------------------------------------------------------------------------

def test_cs_direct_override():
    """CS.Game.Player.Method = function() end should produce a hotfix-style chunk."""
    lines = [
        "CS.Game.UI.HUD.RefreshHealthBar = function(self)",
        "    local hp = self:GetComponent('Health').current",
        "    local max_hp = self:GetComponent('Health').max",
        "    self.healthBar.fillAmount = hp / max_hp",
        "    self.healthText.text = string.format('%d/%d', hp, max_hp)",
        "end",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "hotfix/ui_fix.lua", "hotfix")
    cs_chunks = [c for c in chunks if c.class_name == "HUD"]
    assert len(cs_chunks) == 1
    assert cs_chunks[0].method_name == "RefreshHealthBar"
    assert cs_chunks[0].chunk_type == "method"
    assert "Direct CS override" in cs_chunks[0].doc_comment
    assert cs_chunks[0].namespace == "Game.UI"
    print("[PASS] test_cs_direct_override")


# ---------------------------------------------------------------------------
# Table field functions
# ---------------------------------------------------------------------------

def test_table_field_functions():
    """Table constructors with function fields should produce method chunks."""
    lines = [
        "local AbilityDef = {",
        "    name = 'Fireball',",
        "    damage = 100,",
        "    onCast = function(self, caster, target)",
        "        local dmg = self.damage * caster.spellPower",
        "        target:TakeDamage(dmg)",
        "        Effects.spawn('fire_burst', target.position)",
        "        caster:ConsumeMP(self.manaCost)",
        "    end,",
        "    onHit = function(self, target, damage)",
        "        if target:HasBuff('fire_shield') then",
        "            damage = damage * 0.5",
        "        end",
        "        target:ApplyDOT('burn', damage * 0.2, 5)",
        "        return damage",
        "    end,",
        "}",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "config/abilities.lua", "config")
    ability_chunks = [c for c in chunks if c.class_name == "AbilityDef"]
    method_names = {c.method_name for c in ability_chunks if c.chunk_type == "method"}
    assert "onCast" in method_names
    assert "onHit" in method_names
    print("[PASS] test_table_field_functions")


# ---------------------------------------------------------------------------
# Class-like OOP with class() framework
# ---------------------------------------------------------------------------

def test_class_oop_base_types():
    """class('Name', Base) pattern should propagate base_types to method chunks."""
    lines = [
        "local Player = class('Player', Entity)",
        "",
    ]
    for i in range(10):
        lines.append(f"function Player:ability_{i}(target)")
        for j in range(8):
            lines.append(f"    local v_{j} = target.stat_{j}")
        lines.append("end")
        lines.append("")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "game/player.lua", "game")
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) == 10
    assert all("Entity" in c.base_types for c in method_chunks)
    print("[PASS] test_class_oop_base_types")


# ---------------------------------------------------------------------------
# Constructor detection
# ---------------------------------------------------------------------------

def test_constructor_detection():
    """ctor, new, __init methods should be chunk_type 'constructor'."""
    lines = [
        "local Player = class('Player', Entity)",
        "",
        "function Player:ctor(name, level)",
        "    self.name = name",
        "    self.level = level",
        "    self.hp = 100 * level",
        "    self.mp = 50 * level",
        "end",
        "",
    ]
    # Add enough methods to make it large
    for i in range(15):
        lines.append(f"function Player:method_{i}()")
        for j in range(8):
            lines.append(f"    self.v_{j} = {j}")
        lines.append("end")
        lines.append("")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "game/player.lua", "game")
    ctor_chunks = [c for c in chunks if c.chunk_type == "constructor"]
    assert len(ctor_chunks) == 1
    assert ctor_chunks[0].method_name == "ctor"
    assert "Entity" in ctor_chunks[0].base_types
    print("[PASS] test_constructor_detection")


# ---------------------------------------------------------------------------
# EmmyLua @class annotation
# ---------------------------------------------------------------------------

def test_emmylua_class_annotation():
    """---@class Name : Base should set base_types on method chunks."""
    lines = [
        "---@class BossAI : BaseAI",
        "---Controls boss encounter phases and mechanics",
        "local BossAI = {}",
        "",
    ]
    for i in range(10):
        lines.append(f"function BossAI:phase_{i}()")
        for j in range(8):
            lines.append(f"    self.state_{j} = {j}")
        lines.append("end")
        lines.append("")
    lines.append("return BossAI")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "ai/boss.lua", "ai")
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) == 10
    assert all("BaseAI" in c.base_types for c in method_chunks)
    print("[PASS] test_emmylua_class_annotation")


# ---------------------------------------------------------------------------
# EmmyLua doc comments on functions
# ---------------------------------------------------------------------------

def test_emmylua_function_docs():
    """EmmyLua @param and @return should appear in doc_comment."""
    lines = [
        "local M = {}",
        "",
        "---Calculate effective damage after armor mitigation",
        "---@param raw_damage number",
        "---@param armor number",
        "---@return number",
        "function M.calculateDamage(raw_damage, armor)",
        "    local reduction = armor / (armor + 100)",
        "    local effective = raw_damage * (1 - reduction)",
        "    return math.max(0, effective)",
        "end",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "scripts/combat.lua", "scripts")
    calc_chunks = [c for c in chunks if c.method_name == "calculateDamage"]
    assert len(calc_chunks) == 1
    doc = calc_chunks[0].doc_comment
    assert "effective damage" in doc.lower() or "armor mitigation" in doc.lower()
    assert "@param" in doc or "param" in doc.lower()
    print("[PASS] test_emmylua_function_docs")


# ---------------------------------------------------------------------------
# Class summary generation
# ---------------------------------------------------------------------------

def test_class_summary():
    """Classes with >= 2 methods should get a class_summary chunk."""
    lines = [
        "local Player = class('Player', Entity)",
        "",
    ]
    for i in range(5):
        lines.append(f"function Player:action_{i}()")
        for j in range(8):
            lines.append(f"    self.v_{j} = {j}")
        lines.append("end")
        lines.append("")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "game/player.lua", "game")
    summaries = [c for c in chunks if c.chunk_type == "class_summary"]
    assert len(summaries) >= 1
    player_summary = [s for s in summaries if s.class_name == "Player"]
    assert len(player_summary) == 1
    # Summary should list method signatures
    assert "action_0" in player_summary[0].source
    assert "action_4" in player_summary[0].source
    # Summary should have base types
    assert "Entity" in player_summary[0].base_types
    print("[PASS] test_class_summary")


def test_class_summary_module_table():
    """Module tables with multiple methods should get a class_summary."""
    lines = ["local Utils = {}"]
    for i in range(5):
        lines.append(f"function Utils.helper_{i}(x)")
        for j in range(8):
            lines.append(f"    local v_{j} = x + {j}")
        lines.append("end")
        lines.append("")
    lines.append("return Utils")
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "lib/utils.lua", "lib")
    summaries = [c for c in chunks if c.chunk_type == "class_summary"]
    assert len(summaries) >= 1
    utils_summary = [s for s in summaries if s.class_name == "Utils"]
    assert len(utils_summary) == 1
    assert "local Utils = {}" in utils_summary[0].source
    print("[PASS] test_class_summary_module_table")


# ---------------------------------------------------------------------------
# Mixed game-Lua file (integration test)
# ---------------------------------------------------------------------------

def test_mixed_game_file():
    """A realistic game Lua file mixing multiple patterns."""
    lines = [
        "---@class QuestManager : BaseManager",
        "---Manages quest state, progression, and rewards",
        "local QuestManager = class('QuestManager', BaseManager)",
        "",
        "function QuestManager:ctor()",
        "    self.activeQuests = {}",
        "    self.completedQuests = {}",
        "    self.questLog = {}",
        "end",
        "",
        "function QuestManager:startQuest(questId)",
        "    local quest = QuestDB.get(questId)",
        "    if not quest then return false end",
        "    table.insert(self.activeQuests, quest)",
        "    self:notifyUI('quest_started', quest)",
        "    return true",
        "end",
        "",
        "function QuestManager:completeQuest(questId)",
        "    local quest = self:findActive(questId)",
        "    if not quest then return false end",
        "    quest.completed = true",
        "    self:grantRewards(quest)",
        "    table.insert(self.completedQuests, quest)",
        "    return true",
        "end",
        "",
        "-- Hotfix: fix reward duplication bug in patch 2.1",
        "xlua.hotfix(CS.Game.Quest.RewardSystem, 'GrantReward', function(self, player, reward)",
        "    if player:HasReceivedReward(reward.id) then",
        "        return false",
        "    end",
        "    player:AddItem(reward.itemId, reward.count)",
        "    player:MarkRewardReceived(reward.id)",
        "    return true",
        "end)",
        "",
        "local QUEST_CONFIGS = {",
        "    onAccept = function(self, player)",
        "        player:AddQuestMarker(self.targetNPC)",
        "        player:ShowObjective(self.description)",
        "        DialogManager:Show(self.acceptDialog)",
        "    end,",
        "    onComplete = function(self, player)",
        "        player:RemoveQuestMarker(self.targetNPC)",
        "        player:GrantXP(self.xpReward)",
        "        player:GrantGold(self.goldReward)",
        "    end,",
        "}",
        "",
        "return QuestManager",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "game/quest_manager.lua", "game")

    # Should have QuestManager methods
    qm_methods = [c for c in chunks if c.class_name == "QuestManager" and c.chunk_type == "method"]
    qm_names = {c.method_name for c in qm_methods}
    assert "startQuest" in qm_names
    assert "completeQuest" in qm_names

    # Should have QuestManager constructor
    ctors = [c for c in chunks if c.class_name == "QuestManager" and c.chunk_type == "constructor"]
    assert len(ctors) == 1
    assert ctors[0].method_name == "ctor"

    # Should have base_types from EmmyLua + class()
    all_qm = [c for c in chunks if c.class_name == "QuestManager"]
    for c in all_qm:
        if c.chunk_type != "class_summary":
            assert "BaseManager" in c.base_types, f"Missing base_types on {c.chunk_type} {c.method_name}"

    # Should have xLua hotfix chunk
    hotfix = [c for c in chunks if c.class_name == "RewardSystem"]
    assert len(hotfix) == 1
    assert hotfix[0].method_name == "GrantReward"
    assert "xLua hotfix" in hotfix[0].doc_comment

    # Should have table field function chunks
    config_chunks = [c for c in chunks if c.class_name == "QUEST_CONFIGS"]
    config_names = {c.method_name for c in config_chunks if c.method_name}
    assert "onAccept" in config_names
    assert "onComplete" in config_names

    # Should have class summary for QuestManager
    summaries = [c for c in chunks if c.class_name == "QuestManager" and c.chunk_type == "class_summary"]
    assert len(summaries) == 1
    assert "BaseManager" in summaries[0].base_types

    print("[PASS] test_mixed_game_file")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_hotfix_no_cs_prefix():
    """Hotfix with bare table path (no CS. prefix) should still work."""
    lines = [
        "xlua.hotfix(PlayerController, 'Update', function(self)",
        "    self.speed = 10",
        "    self.velocity = self.direction * self.speed",
        "    self:Move()",
        "end)",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "hotfix/fix.lua", "hotfix")
    hotfix = [c for c in chunks if c.class_name == "PlayerController"]
    assert len(hotfix) == 1
    assert hotfix[0].method_name == "Update"
    print("[PASS] test_hotfix_no_cs_prefix")


def test_constructor_new_pattern():
    """'new' function name should be detected as constructor."""
    lines = [
        "local Weapon = class('Weapon', Item)",
        "",
        "function Weapon:new(name, damage, rarity)",
        "    self.name = name",
        "    self.damage = damage",
        "    self.rarity = rarity",
        "    self.enchantments = {}",
        "end",
        "",
        "function Weapon:getDPS()",
        "    local base = self.damage * self.attackSpeed",
        "    local enchant_bonus = 0",
        "    for _, e in ipairs(self.enchantments) do",
        "        enchant_bonus = enchant_bonus + e.dps",
        "    end",
        "    return base + enchant_bonus",
        "end",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "items/weapon.lua", "items")
    ctors = [c for c in chunks if c.chunk_type == "constructor"]
    assert len(ctors) == 1
    assert ctors[0].method_name == "new"
    assert "Item" in ctors[0].base_types
    print("[PASS] test_constructor_new_pattern")


def test_no_duplicate_class_summaries():
    """A class with only 1 method should NOT get a class_summary."""
    lines = [
        "local M = {}",
        "",
        "function M.onlyMethod(x)",
        "    local result = x * 2",
        "    result = result + 1",
        "    return result",
        "end",
        "",
        "return M",
    ]
    source = _make_large(lines)

    chunks = chunk_file_lua(source, "lib/single.lua", "lib")
    summaries = [c for c in chunks if c.chunk_type == "class_summary"]
    assert len(summaries) == 0
    print("[PASS] test_no_duplicate_class_summaries")


if __name__ == "__main__":
    test_small_file_single_chunk()
    test_top_level_functions()
    test_module_methods()
    test_colon_methods()
    test_local_function()
    test_empty_file_fallback()
    test_mixed_functions_and_methods()
    test_xlua_hotfix_pattern()
    test_namespace_derivation()
    test_preceding_comment_extraction()
    test_hotfix_single_method()
    test_hotfix_batch()
    test_variable_assigned_function()
    test_assignment_method()
    test_cs_direct_override()
    test_table_field_functions()
    test_class_oop_base_types()
    test_constructor_detection()
    test_emmylua_class_annotation()
    test_emmylua_function_docs()
    test_class_summary()
    test_class_summary_module_table()
    test_mixed_game_file()
    test_hotfix_no_cs_prefix()
    test_constructor_new_pattern()
    test_no_duplicate_class_summaries()
    print(f"\n[ALL PASSED] ({26} tests)")
