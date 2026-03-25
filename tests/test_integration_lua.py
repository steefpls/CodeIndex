"""Integration tests for Lua indexing with realistic game data.

End-to-end tests that exercise the full Lua pipeline: chunking -> sidecar
state -> hierarchy/dep graph materialization. Uses synthetic data modeled
after real HoYo-style xLua game codebases.

Run with: PYTHONPATH=. python tests/test_integration_lua.py
"""

import sys
import textwrap
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Realistic game Lua source files
# ---------------------------------------------------------------------------

# A combat system module with EmmyLua annotations and class() OOP
COMBAT_SYSTEM_LUA = textwrap.dedent("""\
    ---@class CombatSystem : GameSystem
    ---Core combat logic: damage calculation, hit detection, status effects
    local CombatSystem = class("CombatSystem", GameSystem)

    function CombatSystem:ctor()
        self.activeEffects = {}
        self.damageLog = {}
        self.critMultiplier = 2.0
        self.armorConstant = 100
    end

    ---Calculate final damage after armor and buffs
    ---@param attacker Entity
    ---@param defender Entity
    ---@param baseDamage number
    ---@return number
    function CombatSystem:calculateDamage(attacker, defender, baseDamage)
        local atk = attacker:GetStat("ATK")
        local def_val = defender:GetStat("DEF")
        local armor = defender:GetStat("ARMOR")

        local raw = baseDamage * (atk / 100)
        local reduction = armor / (armor + self.armorConstant)
        local final = raw * (1 - reduction)

        -- Check for critical hit
        local critRate = attacker:GetStat("CRIT_RATE")
        if math.random() < critRate then
            final = final * self.critMultiplier
            EventBus:Fire("OnCriticalHit", attacker, defender, final)
        end

        -- Apply elemental weakness
        local element = attacker:GetElement()
        local weakness = ElementTable.getWeakness(defender:GetElement(), element)
        final = final * (1 + weakness)

        return math.max(1, math.floor(final))
    end

    ---Apply a status effect to target
    ---@param target Entity
    ---@param effectId string
    ---@param duration number
    function CombatSystem:applyEffect(target, effectId, duration)
        local effect = StatusEffectDB.get(effectId)
        if not effect then
            Logger:Warn("Unknown effect: " .. effectId)
            return false
        end

        -- Check immunity
        if target:HasBuff("immunity_" .. effect.type) then
            return false
        end

        local instance = {
            id = effectId,
            target = target,
            remaining = duration,
            tickRate = effect.tickRate or 1.0,
            onTick = effect.onTick,
        }
        table.insert(self.activeEffects, instance)
        target:AddBuff(effectId, duration)
        EventBus:Fire("OnEffectApplied", target, effectId)
        return true
    end

    ---Process all active effects for this frame
    ---@param dt number delta time
    function CombatSystem:tickEffects(dt)
        local expired = {}
        for i, effect in ipairs(self.activeEffects) do
            effect.remaining = effect.remaining - dt
            if effect.remaining <= 0 then
                table.insert(expired, i)
                effect.target:RemoveBuff(effect.id)
            else
                if effect.onTick then
                    effect.onTick(effect.target, dt)
                end
            end
        end
        -- Remove expired (reverse order)
        for i = #expired, 1, -1 do
            table.remove(self.activeEffects, expired[i])
        end
    end

    function CombatSystem:getActiveBuff(target, effectId)
        for _, effect in ipairs(self.activeEffects) do
            if effect.target == target and effect.id == effectId then
                return effect
            end
        end
        return nil
    end

    function CombatSystem:clearAllEffects(target)
        local remaining = {}
        for _, effect in ipairs(self.activeEffects) do
            if effect.target ~= target then
                table.insert(remaining, effect)
            else
                target:RemoveBuff(effect.id)
            end
        end
        self.activeEffects = remaining
    end

    ---Process a single attack action between two entities
    ---@param attacker Entity
    ---@param defender Entity
    ---@param abilityId string
    function CombatSystem:processAttack(attacker, defender, abilityId)
        local ability = AbilityDB.get(abilityId)
        if not ability then
            Logger:Error("Unknown ability: " .. abilityId)
            return nil
        end

        -- Check if ability is on cooldown
        if attacker:IsOnCooldown(abilityId) then
            EventBus:Fire("OnAbilityCooldown", attacker, abilityId)
            return nil
        end

        -- Check resource cost
        local cost = ability.manaCost or 0
        if attacker:GetStat("MP") < cost then
            EventBus:Fire("OnInsufficientResource", attacker, abilityId, cost)
            return nil
        end

        -- Calculate and apply damage
        local damage = self:calculateDamage(attacker, defender, ability.baseDamage)
        defender:TakeDamage(damage)
        attacker:ConsumeMP(cost)
        attacker:StartCooldown(abilityId, ability.cooldown)

        -- Log the attack
        table.insert(self.damageLog, {
            timestamp = os.time(),
            attacker = attacker:GetId(),
            defender = defender:GetId(),
            ability = abilityId,
            damage = damage,
        })

        -- Apply ability effects
        if ability.effects then
            for _, fx in ipairs(ability.effects) do
                self:applyEffect(defender, fx.id, fx.duration)
            end
        end

        EventBus:Fire("OnAttackComplete", attacker, defender, damage, abilityId)
        return damage
    end

    ---Get combat statistics for an entity
    ---@param entityId string
    ---@return table
    function CombatSystem:getStats(entityId)
        local totalDamage = 0
        local totalHits = 0
        local critCount = 0
        for _, entry in ipairs(self.damageLog) do
            if entry.attacker == entityId then
                totalDamage = totalDamage + entry.damage
                totalHits = totalHits + 1
            end
        end
        return {
            totalDamage = totalDamage,
            totalHits = totalHits,
            averageDamage = totalHits > 0 and (totalDamage / totalHits) or 0,
        }
    end

    return CombatSystem
""")

# xLua hotfix patching multiple methods on a C# class
HOTFIX_GACHA_LUA = textwrap.dedent("""\
    ---Hotfix for gacha pull rate display bug (patch 2.4.1)
    ---Shows wrong rate-up percentage on limited banner

    local util = require("xlua.util")
    local json = require("cjson")

    -- Fix: CalculatePullRate was using base rate instead of rate-up rate
    xlua.hotfix(CS.Game.Gacha.GachaManager, 'CalculatePullRate', function(self, bannerId, itemId)
        local banner = self:GetBanner(bannerId)
        if not banner then
            return 0
        end

        local isRateUp = false
        for _, upId in ipairs(banner.rateUpItems) do
            if upId == itemId then
                isRateUp = true
                break
            end
        end

        if isRateUp then
            return banner.rateUpChance
        end
        return banner.baseRate / banner.poolSize
    end)

    -- Fix: PityCounter wasn't resetting on SSR pull
    xlua.hotfix(CS.Game.Gacha.GachaManager, 'ProcessPull', function(self, bannerId)
        local banner = self:GetBanner(bannerId)
        local pity = self:GetPityCount(bannerId)
        local rate = self:CalculatePullRate(bannerId, banner.currentRateUp)

        -- Soft pity: increase rate after 74 pulls
        if pity >= 74 then
            rate = rate + (pity - 73) * 0.06
        end

        local result = math.random() < math.min(rate, 1.0)
        if result then
            self:ResetPityCounter(bannerId)  -- THIS WAS MISSING
            self:GrantItem(banner.currentRateUp)
            EventBus:Fire("OnSSRPull", bannerId, banner.currentRateUp)
        else
            self:IncrementPity(bannerId)
        end
        return result
    end)

    -- Fix: History display not showing correct timestamps
    xlua.hotfix(CS.Game.Gacha.GachaManager, 'GetPullHistory', function(self, bannerId, count)
        local history = self:GetRawHistory(bannerId)
        if not history then return {} end

        local result = {}
        local start = math.max(1, #history - count + 1)
        for i = start, #history do
            local entry = history[i]
            table.insert(result, {
                itemId = entry.itemId,
                rarity = entry.rarity,
                timestamp = entry.timestamp,
                pityCount = entry.pityAtPull,
                wasRateUp = entry.wasRateUp or false,
            })
        end
        return result
    end)

    -- Fix: Guarantee tracking across banners
    xlua.hotfix(CS.Game.Gacha.GachaManager, 'CheckGuarantee', function(self, bannerId)
        local banner = self:GetBanner(bannerId)
        if not banner then return false end

        local lastSSR = self:GetLastSSRPull(bannerId)
        if lastSSR and not lastSSR.wasRateUp then
            return true  -- next SSR is guaranteed rate-up
        end
        return false
    end)

    -- Fix: Banner expiry check was using server time incorrectly
    xlua.hotfix(CS.Game.Gacha.GachaManager, 'IsBannerActive', function(self, bannerId)
        local banner = self:GetBanner(bannerId)
        if not banner then return false end

        local serverTime = TimeManager:GetServerTime()
        return serverTime >= banner.startTime and serverTime < banner.endTime
    end)

    -- Batch fix for display-related methods
    xlua.hotfix(CS.Game.UI.BannerDisplay, {
        RefreshRateText = function(self)
            local banner = GachaManager:GetActiveBanner()
            if not banner then return end
            local rate = GachaManager:CalculatePullRate(banner.id, banner.currentRateUp)
            self.rateText.text = string.format("%.1f%%", rate * 100)
            self.pityText.text = string.format("Pity: %d/90", GachaManager:GetPityCount(banner.id))
        end,
        RefreshBannerArt = function(self)
            local banner = GachaManager:GetActiveBanner()
            if not banner then return end
            self.bannerImage.sprite = ResourceManager:LoadSprite(banner.artPath)
            self.characterName.text = ItemDB:GetName(banner.currentRateUp)
            self.elementIcon.sprite = ResourceManager:LoadSprite(
                ElementTable.getIcon(ItemDB:GetElement(banner.currentRateUp))
            )
        end,
    })
""")

# Ability definition table with function fields
ABILITY_FIREBALL_LUA = textwrap.dedent("""\
    ---@class FireballAbility
    ---Ranged fire ability: launches a projectile that explodes on contact
    local FireballAbility = {
        id = "ability_fireball",
        name = "Fireball",
        element = "fire",
        baseDamage = 150,
        cooldown = 8.0,
        manaCost = 40,
        aoeRadius = 3.0,

        onCast = function(self, caster, targetPos)
            local dir = (targetPos - caster:GetPosition()):Normalized()
            local projectile = ProjectileManager:Spawn("fireball_vfx", caster:GetPosition(), dir)
            projectile.speed = 15
            projectile.damage = self.baseDamage * caster:GetStat("SPELL_POWER") / 100
            projectile.element = self.element
            projectile.onHit = function(target)
                CombatSystem:dealDamage(caster, target, projectile.damage, self.element)
                -- AoE explosion
                local nearby = PhysicsQuery:SphereOverlap(target:GetPosition(), self.aoeRadius)
                for _, entity in ipairs(nearby) do
                    if entity ~= target and entity:IsEnemy(caster) then
                        local splashDmg = projectile.damage * 0.5
                        CombatSystem:dealDamage(caster, entity, splashDmg, self.element)
                    end
                end
            end
            SoundManager:Play("fireball_cast")
            caster:ConsumeMP(self.manaCost)
            caster:StartCooldown(self.id, self.cooldown)
        end,

        onUpgrade = function(self, level)
            self.baseDamage = 150 + level * 25
            self.aoeRadius = 3.0 + level * 0.3
            if level >= 5 then
                self.manaCost = 35  -- reduced cost at max level
            end
        end,

        getTooltip = function(self, caster)
            local dmg = self.baseDamage * caster:GetStat("SPELL_POWER") / 100
            return string.format(
                "Launches a fireball dealing %d fire damage.\\n"
                .. "Explodes on contact for %d AoE damage (%.1fm radius).\\n"
                .. "Cooldown: %.1fs | Cost: %d MP",
                dmg, dmg * 0.5, self.aoeRadius, self.cooldown, self.manaCost
            )
        end,

        canCast = function(self, caster)
            if caster:IsOnCooldown(self.id) then
                return false, "On cooldown"
            end
            if caster:GetStat("MP") < self.manaCost then
                return false, "Not enough MP"
            end
            if caster:HasDebuff("silence") then
                return false, "Silenced"
            end
            if caster:GetStat("HP") <= 0 then
                return false, "Dead"
            end
            return true, nil
        end,

        onLevelUp = function(self, caster, newLevel)
            self:onUpgrade(newLevel)
            -- Update cached stats
            local newDmg = self.baseDamage * caster:GetStat("SPELL_POWER") / 100
            Logger:Info(string.format(
                "Fireball leveled to %d: %d base damage, %.1fm AoE, %d MP cost",
                newLevel, self.baseDamage, self.aoeRadius, self.manaCost
            ))
            -- Refresh tooltip if ability panel is open
            if UIManager:IsPanelOpen("AbilityPanel") then
                UIManager:RefreshPanel("AbilityPanel")
            end
            EventBus:Fire("OnAbilityUpgrade", caster, self.id, newLevel)
        end,

        serialize = function(self)
            return {
                id = self.id,
                name = self.name,
                element = self.element,
                baseDamage = self.baseDamage,
                cooldown = self.cooldown,
                manaCost = self.manaCost,
                aoeRadius = self.aoeRadius,
            }
        end,

        deserialize = function(self, data)
            self.baseDamage = data.baseDamage or self.baseDamage
            self.cooldown = data.cooldown or self.cooldown
            self.manaCost = data.manaCost or self.manaCost
            self.aoeRadius = data.aoeRadius or self.aoeRadius
        end,

        getPreviewDamage = function(self, caster, targetDef)
            local spellPower = caster:GetStat("SPELL_POWER")
            local baseDmg = self.baseDamage * spellPower / 100
            local armor = targetDef or 0
            local reduction = armor / (armor + 100)
            local finalDmg = baseDmg * (1 - reduction)
            local critDmg = finalDmg * 2.0
            return {
                minimum = math.floor(finalDmg * 0.9),
                maximum = math.floor(finalDmg * 1.1),
                critical = math.floor(critDmg),
                aoeMinimum = math.floor(finalDmg * 0.45),
                aoeMaximum = math.floor(finalDmg * 0.55),
            }
        end,

        getAnimationData = function(self, caster)
            local castTime = 0.8
            local projectileTime = 1.2
            local explosionTime = 0.3
            return {
                castAnimation = "spell_cast_fire",
                castDuration = castTime,
                projectileVfx = "fireball_projectile",
                projectileDuration = projectileTime,
                impactVfx = "fireball_explosion",
                impactDuration = explosionTime,
                impactSound = "explosion_fire_medium",
                totalDuration = castTime + projectileTime + explosionTime,
                cameraShake = { intensity = 0.3, duration = 0.2 },
            }
        end,

        getAIWeight = function(self, caster, target, context)
            local distance = (target:GetPosition() - caster:GetPosition()):Magnitude()
            local canUse, reason = self:canCast(caster)
            if not canUse then return 0 end

            local weight = 50  -- base weight
            -- Prefer if target is weak to fire
            local weakness = ElementTable.getWeakness(target:GetElement(), self.element)
            weight = weight + weakness * 40

            -- Prefer AoE when multiple enemies are grouped
            local nearby = PhysicsQuery:SphereOverlap(target:GetPosition(), self.aoeRadius)
            local enemyCount = 0
            for _, e in ipairs(nearby) do
                if e:IsEnemy(caster) then enemyCount = enemyCount + 1 end
            end
            weight = weight + (enemyCount - 1) * 15

            return math.max(0, weight)
        end,
    }

    return FireballAbility
""")

# Small utility module (should produce single whole_class chunk)
ELEMENT_TABLE_LUA = textwrap.dedent("""\
    ---@class ElementTable
    ---Elemental weakness/resistance lookup
    local ElementTable = {}

    local WEAKNESS_MAP = {
        fire =  { weak = "ice",   resist = "fire"  },
        ice =   { weak = "fire",  resist = "ice"   },
        wind =  { weak = "earth", resist = "wind"  },
        earth = { weak = "wind",  resist = "earth" },
    }

    function ElementTable.getWeakness(defenderElement, attackerElement)
        local entry = WEAKNESS_MAP[defenderElement]
        if not entry then return 0 end
        if entry.weak == attackerElement then return 0.5 end
        if entry.resist == attackerElement then return -0.3 end
        return 0
    end

    function ElementTable.getIcon(element)
        return "icons/element_" .. (element or "none")
    end

    return ElementTable
""")


class TestLuaChunkingRealisticData(unittest.TestCase):
    """Test Lua chunker against realistic game codebase files."""

    def test_combat_system_chunking(self):
        """Large OOP module with EmmyLua should produce class_summary + methods."""
        from src.indexer.chunker_lua import chunk_file_lua
        chunks = chunk_file_lua(
            COMBAT_SYSTEM_LUA.encode(), "game/combat/combat_system.lua", "combat",
        )

        # Should have a class summary
        summaries = [c for c in chunks if c.chunk_type == "class_summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].class_name, "CombatSystem")
        self.assertIn("GameSystem", summaries[0].base_types)

        # Should have constructor
        ctors = [c for c in chunks if c.chunk_type == "constructor"]
        self.assertEqual(len(ctors), 1)
        self.assertEqual(ctors[0].method_name, "ctor")
        self.assertIn("GameSystem", ctors[0].base_types)

        # Should have all methods
        methods = [c for c in chunks if c.chunk_type == "method"]
        method_names = {c.method_name for c in methods}
        self.assertIn("calculateDamage", method_names)
        self.assertIn("applyEffect", method_names)
        self.assertIn("tickEffects", method_names)
        self.assertIn("getActiveBuff", method_names)
        self.assertIn("clearAllEffects", method_names)
        self.assertIn("processAttack", method_names)
        self.assertIn("getStats", method_names)

        # EmmyLua doc comments should be captured
        calc = [c for c in methods if c.method_name == "calculateDamage"][0]
        self.assertIn("damage", calc.doc_comment.lower())
        self.assertIn("@param", calc.doc_comment)

    def test_hotfix_gacha_chunking(self):
        """xLua hotfix file with single + batch forms should extract all methods."""
        from src.indexer.chunker_lua import chunk_file_lua
        chunks = chunk_file_lua(
            HOTFIX_GACHA_LUA.encode(), "hotfix/gacha_fix.lua", "hotfix",
        )

        # Single-method hotfixes for GachaManager
        gacha_methods = [c for c in chunks
                         if c.class_name == "GachaManager" and c.chunk_type == "method"]
        gacha_names = {c.method_name for c in gacha_methods}
        self.assertIn("CalculatePullRate", gacha_names)
        self.assertIn("ProcessPull", gacha_names)
        self.assertIn("GetPullHistory", gacha_names)
        self.assertIn("CheckGuarantee", gacha_names)
        self.assertIn("IsBannerActive", gacha_names)
        # Namespace should be extracted from CS.Game.Gacha
        for m in gacha_methods:
            self.assertEqual(m.namespace, "Game.Gacha")
            self.assertIn("xLua hotfix", m.doc_comment)

        # Batch hotfix for BannerDisplay
        display_methods = [c for c in chunks
                           if c.class_name == "BannerDisplay" and c.chunk_type == "method"]
        display_names = {c.method_name for c in display_methods}
        self.assertIn("RefreshRateText", display_names)
        self.assertIn("RefreshBannerArt", display_names)
        for m in display_methods:
            self.assertEqual(m.namespace, "Game.UI")

    def test_ability_table_chunking(self):
        """Table constructor with function fields should produce method chunks."""
        from src.indexer.chunker_lua import chunk_file_lua
        chunks = chunk_file_lua(
            ABILITY_FIREBALL_LUA.encode(), "config/abilities/fireball.lua", "config",
        )

        # Should have method chunks for function fields
        methods = [c for c in chunks if c.chunk_type == "method"]
        method_names = {c.method_name for c in methods}
        self.assertIn("onCast", method_names)
        self.assertIn("onUpgrade", method_names)
        self.assertIn("getTooltip", method_names)
        self.assertIn("canCast", method_names)
        self.assertIn("serialize", method_names)

        # All methods should have class_name = table name
        for m in methods:
            self.assertEqual(m.class_name, "FireballAbility")

    def test_small_module_single_chunk(self):
        """Small utility module should produce single whole_class chunk."""
        from src.indexer.chunker_lua import chunk_file_lua
        chunks = chunk_file_lua(
            ELEMENT_TABLE_LUA.encode(), "game/data/element_table.lua", "data",
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "whole_class")
        self.assertEqual(chunks[0].class_name, "ElementTable")


class TestLuaSidecarIntegration(unittest.TestCase):
    """Test that Lua chunks feed into sidecar builders correctly."""

    def test_hierarchy_from_combat_system(self):
        """CombatSystem extending GameSystem should appear in type hierarchy."""
        from src.indexer.chunker_lua import chunk_file_lua
        from src.indexer.hierarchy_builder import build_type_hierarchy

        chunks = chunk_file_lua(
            COMBAT_SYSTEM_LUA.encode(), "game/combat/combat_system.lua", "combat",
        )

        # Build hierarchy records (same logic as pipeline.py lines 311-316)
        records = []
        for c in chunks:
            if c.class_name and c.base_types and c.chunk_type in ("class_summary", "whole_class"):
                records.append((
                    c.chunk_type, c.class_name, c.file_path,
                    c.module, c.namespace, list(c.base_types),
                ))

        hierarchy = build_type_hierarchy(records)
        self.assertIn("GameSystem", hierarchy)
        impls = [e["class"] for e in hierarchy["GameSystem"]]
        self.assertIn("CombatSystem", impls)

    def test_dep_graph_from_combat_system(self):
        """CombatSystem should have dep graph entries for referenced types."""
        from src.indexer.chunker_lua import chunk_file_lua
        from src.indexer.dep_graph_builder import extract_type_candidates, CODE_CHUNK_TYPES

        chunks = chunk_file_lua(
            COMBAT_SYSTEM_LUA.encode(), "game/combat/combat_system.lua", "combat",
        )

        # Collect all type candidates from all code chunks (pipeline logic)
        all_candidates = set()
        for c in chunks:
            if c.chunk_type in CODE_CHUNK_TYPES:
                all_candidates.update(extract_type_candidates(c.source))

        # Should find referenced types from the combat system code
        self.assertIn("EventBus", all_candidates)
        self.assertIn("StatusEffectDB", all_candidates)
        self.assertIn("ElementTable", all_candidates)
        self.assertIn("Logger", all_candidates)

    def test_dep_graph_from_hotfix(self):
        """Hotfix file should extract CS.* type references."""
        from src.indexer.chunker_lua import chunk_file_lua
        from src.indexer.dep_graph_builder import extract_type_candidates, CODE_CHUNK_TYPES

        chunks = chunk_file_lua(
            HOTFIX_GACHA_LUA.encode(), "hotfix/gacha_fix.lua", "hotfix",
        )

        all_candidates = set()
        for c in chunks:
            if c.chunk_type in CODE_CHUNK_TYPES:
                all_candidates.update(extract_type_candidates(c.source))

        # Should find C# types from CS.* references and require()
        self.assertIn("GachaManager", all_candidates)
        self.assertIn("BannerDisplay", all_candidates)
        self.assertIn("ResourceManager", all_candidates)
        self.assertIn("ItemDB", all_candidates)


class TestLuaEmbeddingTextQuality(unittest.TestCase):
    """Verify that Lua chunk embedding text is search-friendly."""

    def test_hotfix_embedding_contains_cs_class(self):
        """Hotfix chunk embedding should include the C# class name for search."""
        from src.indexer.chunker_lua import chunk_file_lua
        chunks = chunk_file_lua(
            HOTFIX_GACHA_LUA.encode(), "hotfix/gacha_fix.lua", "hotfix",
        )

        calc_rate = [c for c in chunks if c.method_name == "CalculatePullRate"][0]
        embed = calc_rate.embedding_text
        # Embedding text header should include the C# class name
        self.assertIn("GachaManager", embed)
        # Should include the C# namespace context
        self.assertIn("Game.Gacha", embed)
        # Should include the hotfix doc comment
        self.assertIn("xLua hotfix", embed)

    def test_oop_class_embedding_includes_base_types(self):
        """OOP class chunks should include base type in embedding header."""
        from src.indexer.chunker_lua import chunk_file_lua
        chunks = chunk_file_lua(
            COMBAT_SYSTEM_LUA.encode(), "game/combat/combat_system.lua", "combat",
        )

        summary = [c for c in chunks if c.chunk_type == "class_summary"][0]
        embed = summary.embedding_text
        # Should include class name and base type in header
        self.assertIn("CombatSystem", embed)
        self.assertIn("GameSystem", embed)

    def test_table_field_embedding_includes_table_name(self):
        """Table field function chunks should reference their parent table."""
        from src.indexer.chunker_lua import chunk_file_lua
        chunks = chunk_file_lua(
            ABILITY_FIREBALL_LUA.encode(), "config/abilities/fireball.lua", "config",
        )

        on_cast = [c for c in chunks if c.method_name == "onCast"][0]
        embed = on_cast.embedding_text
        self.assertIn("FireballAbility", embed)


if __name__ == "__main__":
    unittest.main()
