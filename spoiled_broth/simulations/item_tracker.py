"""
Item Tracker - Tracks items with IDs and collaboration during simulation.

Manages item lifecycle (creation, pickup, drop, cutting, assembly, delivery)
and tracks which agents touched each item for collaboration analysis.

GAME STATE SYNCHRONIZATION:
============================
The tracker now syncs with actual game state before each action to prevent
desynchronization. Instead of guessing from action names, it reads:
- game.gameObjects[agent_id].item - what each agent is holding
- game.grid.tiles[x][y].item - what's on each counter

This eliminates the shadow tracker problem where parallel state gets out of sync.

ITEM LINEAGE:
=============
Maintains complete parent-child relationships:
- tomato → tomato_cut (via cutting)
- plate + tomato_cut → tomato_salad (via assembly)

All items track their origins dictionary to maintain the full chain.

Author: Samuel Lozano
"""

import csv
from typing import Dict, List, Optional, Tuple, Any


class ItemTracker:
    """Track items with IDs and collaboration during simulation."""
    
    def __init__(self):
        self.item_counters = {'tomato': 0, 'plate': 0, 'tomato_cut': 0, 'tomato_salad': 0}
        self.items = {}  # item_id -> {type, touched_by, origins, roles}
        self.recorded_items: List[Dict[str, Any]] = []
        self._creation_counter = 0
        self._recorded_item_counter = 0
        self.agent_holding = {}  # agent_id -> item_id
        self.counter_items = {}  # (x, y) -> item_id
        
    def create_item(
        self,
        item_type: str,
        agent_id: str,
        origins: Dict = None,
        tick: Optional[int] = None,
        second: Optional[float] = None,
        source: str = '',
    ) -> str:
        """Create new tracked item."""
        self.item_counters[item_type] += 1
        item_id = f"{item_type}_{self.item_counters[item_type]}"
        self._creation_counter += 1

        # Sanitise origins: strip None / empty-string values so downstream
        # code can always treat a missing key as "unknown".
        clean_origins = {k: v for k, v in (origins or {}).items() if v}

        self.items[item_id] = {
            'type': item_type,
            'creation_index': self._creation_counter,
            'created_by': agent_id,
            'created_tick': tick if tick is not None else '',
            'created_second': f"{second:.3f}" if second is not None else '',
            'created_source': source,
            'touched_by': [agent_id] if agent_id else [],
            'last_touched': agent_id,
            'origins': clean_origins,
            'counters_used': [],
            'tomato_id': '',
            'plate_id': '',
            'tomato_cut_id': '',
            'tomato_salad_id': '',
            'who_picked_tomato': '',
            'who_picked_plate': '',
            'who_cut': '',
            'who_assembled': '',
            'who_delivered': ''
        }

        # Set roles based on type and propagate from origin items
        if item_type == 'tomato' and agent_id:
            self.items[item_id]['who_picked_tomato'] = agent_id
        elif item_type == 'plate' and agent_id:
            self.items[item_id]['who_picked_plate'] = agent_id
        elif item_type == 'tomato_cut' and agent_id:
            self.items[item_id]['who_cut'] = agent_id
            # Inherit who_picked_tomato from the source tomato (if tracked)
            tomato_origin = clean_origins.get('tomato_id')
            if tomato_origin and tomato_origin in self.items:
                self.items[item_id]['tomato_id'] = tomato_origin
                self.items[item_id]['who_picked_tomato'] = self.items[tomato_origin]['who_picked_tomato']
        elif item_type == 'tomato_salad' and agent_id:
            self.items[item_id]['who_assembled'] = agent_id
            # Inherit all per-step roles from origin items
            for origin_key in ['tomato_cut_id', 'plate_id']:
                origin_id = clean_origins.get(origin_key)
                if origin_id and origin_id in self.items:
                    self.items[item_id][origin_key] = origin_id
                    origin = self.items[origin_id]
                    for role in ['who_picked_tomato', 'who_picked_plate', 'who_cut']:
                        if origin.get(role) and not self.items[item_id][role]:
                            self.items[item_id][role] = origin[role]
        return item_id

    def record_delivered_item(
        self,
        agent_id: str,
        salad_item_id: Optional[str],
        tick: Optional[int] = None,
        second: Optional[float] = None,
    ) -> Optional[str]:
        """Record a delivered item only for export, without mutating live game state."""
        if not salad_item_id or salad_item_id not in self.items:
            return None

        salad_data = self.get_item_data(salad_item_id)

        self._recorded_item_counter += 1
        delivered_id = f"tomato_delivered_{self._recorded_item_counter}"
        self._creation_counter += 1

        delivered_touched_list = [agent_id] if agent_id else []
        history_touched_list = [
            value for value in salad_data.get('touched_list_history', '').split(';')
            if value
        ]
        if not history_touched_list:
            history_touched_list = [
                value for value in salad_data.get('touched_list', '').split(';')
                if value
            ]
        if agent_id and agent_id not in history_touched_list:
            history_touched_list.append(agent_id)

        self.recorded_items.append({
            'creation_index': self._creation_counter,
            'item_id': delivered_id,
            'item_type': 'tomato_delivered',
            'created_by': agent_id,
            'created_tick': tick if tick is not None else '',
            'created_second': f"{second:.3f}" if second is not None else '',
            'created_source': 'delivery',
            'last_touched': agent_id,
            'touched_list': ';'.join(delivered_touched_list),
            'touched_list_history': ';'.join(history_touched_list),
            'tomato_id': salad_data.get('tomato_id', ''),
            'plate_id': salad_data.get('plate_id', ''),
            'tomato_cut_id': salad_data.get('tomato_cut_id', ''),
            'tomato_salad_id': salad_item_id,
            'tomato_delivered_id': delivered_id,
            'who_picked_tomato': salad_data.get('who_picked_tomato', ''),
            'who_picked_plate': salad_data.get('who_picked_plate', ''),
            'who_cutted': salad_data.get('who_cutted', ''),
            'who_assembled': salad_data.get('who_assembled', ''),
            'who_delivered': agent_id,
            'number_of_counters_used': salad_data.get('number_of_counters_used', 0),
        })
        return delivered_id
    
    def get_item_data(self, item_id: str) -> Dict:
        """Get comprehensive item data for CSV output."""
        if not item_id or item_id not in self.items:
            return self._empty_data()
        
        item = self.items[item_id]
        
        # Build ID lineage — always use '' not None for missing links
        ids = {
            'tomato_id': item.get('tomato_id', ''),
            'plate_id': item.get('plate_id', ''),
            'tomato_cut_id': item.get('tomato_cut_id', ''),
            'tomato_salad_id': item.get('tomato_salad_id', ''),
        }
        if item['type'] == 'tomato':
            ids['tomato_id'] = item_id
        elif item['type'] == 'plate':
            ids['plate_id'] = item_id
        elif item['type'] == 'tomato_cut':
            ids['tomato_cut_id'] = item_id
            ids['tomato_id'] = item['origins'].get('tomato_id') or ''
        elif item['type'] == 'tomato_salad':
            ids['tomato_salad_id'] = item_id
            ids['tomato_cut_id'] = item['origins'].get('tomato_cut_id') or ''
            ids['plate_id'] = item['origins'].get('plate_id') or ''
            # Walk back through tomato_cut to find the source tomato
            tc_id = ids['tomato_cut_id']
            if tc_id and tc_id in self.items:
                ids['tomato_id'] = self.items[tc_id]['origins'].get('tomato_id') or ''

        who_picked_tomato = item.get('who_picked_tomato', '')
        who_picked_plate = item.get('who_picked_plate', '')
        who_cutted = item.get('who_cut', '')
        who_assembled = item.get('who_assembled', '')
        who_delivered = item.get('who_delivered', '')

        tomato_id = ids.get('tomato_id', '')
        plate_id = ids.get('plate_id', '')
        tomato_cut_id = ids.get('tomato_cut_id', '')
        tomato_salad_id = ids.get('tomato_salad_id', '')

        if tomato_cut_id and tomato_cut_id in self.items:
            cut_item = self.items[tomato_cut_id]
            if not who_cutted:
                who_cutted = cut_item.get('who_cut', '')
            if not who_picked_tomato:
                who_picked_tomato = cut_item.get('who_picked_tomato', '')
            if not tomato_id:
                tomato_id = cut_item.get('origins', {}).get('tomato_id', '')

        if tomato_id and tomato_id in self.items and not who_picked_tomato:
            who_picked_tomato = self.items[tomato_id].get('who_picked_tomato', '')

        if plate_id and plate_id in self.items and not who_picked_plate:
            who_picked_plate = self.items[plate_id].get('who_picked_plate', '')

        if tomato_salad_id and tomato_salad_id in self.items and not who_assembled:
            who_assembled = (
                self.items[tomato_salad_id].get('who_assembled', '')
                or self.items[tomato_salad_id].get('created_by', '')
            )
        elif item['type'] == 'tomato_salad' and not who_assembled:
            who_assembled = item.get('created_by', '')

        ids['tomato_id'] = tomato_id or ''
        ids['plate_id'] = plate_id or ''
        ids['tomato_cut_id'] = tomato_cut_id or ''
        ids['tomato_salad_id'] = tomato_salad_id or ''

        # Collaboration tracking:
        # - item_touched are touches on this concrete item only
        # - history_touched are touches across full lineage (origins + this item)
        item_touched_list = [value for value in item.get('touched_by', []) if value]
        item_touched = set(item_touched_list)
        history_touched_list = self._get_lineage_touched_list(item_id)
        history_touched = set(history_touched_list)
        
        return {
            **ids,
            'item_id': item_id,
            'item_type': item['type'],
            'creation_index': item.get('creation_index', ''),
            'created_by': item.get('created_by', ''),
            'created_tick': item.get('created_tick', ''),
            'created_second': item.get('created_second', ''),
            'created_source': item.get('created_source', ''),
            'last_touched': item['last_touched'],
            'touched_list': ';'.join(item_touched_list),
            'touched_list_history': ';'.join(history_touched_list),
            'is_item_collaboration': len(item_touched) > 1,
            'is_history_collaboration': len(history_touched) > 1,
            'who_picked_tomato': who_picked_tomato,
            'who_picked_plate': who_picked_plate,
            'who_cutted': who_cutted,
            'who_assembled': who_assembled,
            'who_delivered': who_delivered,
            'number_of_counters_used': len(item['counters_used'])
        }

    def _get_lineage_touched_list(self, item_id: str) -> List[str]:
        """Return ordered unique touched agents across full item lineage."""
        if item_id not in self.items:
            return []

        # Build lineage item IDs ordered by creation index: origins first, then item.
        lineage_ids = [oid for oid in self._get_all_origins(item_id) if oid in self.items]
        lineage_ids.append(item_id)
        lineage_ids = sorted(
            set(lineage_ids),
            key=lambda iid: self.items.get(iid, {}).get('creation_index', 0)
        )

        touched_history: List[str] = []
        seen_agents = set()
        for iid in lineage_ids:
            for agent_id in self.items[iid].get('touched_by', []):
                if agent_id and agent_id not in seen_agents:
                    seen_agents.add(agent_id)
                    touched_history.append(agent_id)

        return touched_history

    def export_items(self, csv_path):
        """Write the current item ledger to items.csv."""
        header = [
            'item_id', 'item_type', 'creation_index', 'created_by', 'created_tick', 'created_second', 'created_source',
            'last_touched', 'touched_list', 'touched_list_history',
            'tomato_id', 'plate_id', 'tomato_cut_id', 'tomato_salad_id', 'tomato_delivered_id',
            'who_picked_tomato', 'who_picked_plate', 'who_cutted', 'who_assembled', 'who_delivered',
            'number_of_counters_used',
        ]

        rows = []
        for item_id, item in self.items.items():
            data = self.get_item_data(item_id)
            rows.append((item.get('creation_index', 0), [
                data.get('item_id', item_id),
                data.get('item_type', item.get('type', '')),
                data.get('creation_index', ''),
                data.get('created_by', ''),
                data.get('created_tick', ''),
                data.get('created_second', ''),
                data.get('created_source', ''),
                data.get('last_touched', ''),
                data.get('touched_list', ''),
                data.get('touched_list_history', ''),
                data.get('tomato_id', ''),
                data.get('plate_id', ''),
                data.get('tomato_cut_id', ''),
                data.get('tomato_salad_id', ''),
                data.get('tomato_delivered_id', ''),
                data.get('who_picked_tomato', ''),
                data.get('who_picked_plate', ''),
                data.get('who_cutted', ''),
                data.get('who_assembled', ''),
                data.get('who_delivered', ''),
                data.get('number_of_counters_used', 0),
            ]))

        for record in self.recorded_items:
            rows.append((record.get('creation_index', 0), [
                record.get('item_id', ''),
                record.get('item_type', ''),
                record.get('creation_index', ''),
                record.get('created_by', ''),
                record.get('created_tick', ''),
                record.get('created_second', ''),
                record.get('created_source', ''),
                record.get('last_touched', ''),
                record.get('touched_list', ''),
                record.get('touched_list_history', ''),
                record.get('tomato_id', ''),
                record.get('plate_id', ''),
                record.get('tomato_cut_id', ''),
                record.get('tomato_salad_id', ''),
                record.get('tomato_delivered_id', ''),
                record.get('who_picked_tomato', ''),
                record.get('who_picked_plate', ''),
                record.get('who_cutted', ''),
                record.get('who_assembled', ''),
                record.get('who_delivered', ''),
                record.get('number_of_counters_used', 0),
            ]))

        rows.sort(key=lambda entry: entry[0])

        with open(csv_path, 'w', newline='') as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(header)
            for _, row in rows:
                writer.writerow(row)

    def backfill_item_origins(
        self,
        item_id: Optional[str],
        origins: Optional[Dict[str, str]],
        agent_id: str = '',
    ) -> bool:
        """Merge inferred origins into an existing item and refresh lineage metadata."""
        if not item_id or item_id not in self.items or not origins:
            return False

        item = self.items[item_id]
        clean_origins = {
            key: value
            for key, value in origins.items()
            if value and value in self.items
        }
        if not clean_origins:
            return False

        updated = False
        for key, value in clean_origins.items():
            if not item['origins'].get(key):
                item['origins'][key] = value
                updated = True

        if item['type'] == 'tomato_cut':
            tomato_origin = item['origins'].get('tomato_id')
            if tomato_origin and tomato_origin in self.items:
                if item.get('tomato_id') != tomato_origin:
                    item['tomato_id'] = tomato_origin
                    updated = True
                picked_by = self.items[tomato_origin].get('who_picked_tomato', '')
                if picked_by and not item.get('who_picked_tomato'):
                    item['who_picked_tomato'] = picked_by
                    updated = True
            if agent_id and not item.get('who_cut'):
                item['who_cut'] = agent_id
                updated = True

        elif item['type'] == 'tomato_salad':
            tomato_cut_id = item['origins'].get('tomato_cut_id')
            plate_id = item['origins'].get('plate_id')

            if tomato_cut_id and item.get('tomato_cut_id') != tomato_cut_id:
                item['tomato_cut_id'] = tomato_cut_id
                updated = True
            if plate_id and item.get('plate_id') != plate_id:
                item['plate_id'] = plate_id
                updated = True

            if tomato_cut_id and tomato_cut_id in self.items:
                source_tomato_id = (
                    self.items[tomato_cut_id]['origins'].get('tomato_id')
                    or self.items[tomato_cut_id].get('tomato_id', '')
                )
                if source_tomato_id and not item.get('tomato_id'):
                    item['tomato_id'] = source_tomato_id
                    updated = True

            for origin_id in [tomato_cut_id, plate_id]:
                if not origin_id or origin_id not in self.items:
                    continue
                origin = self.items[origin_id]
                for role in ['who_picked_tomato', 'who_picked_plate', 'who_cut']:
                    if origin.get(role) and not item.get(role):
                        item[role] = origin[role]
                        updated = True

            if agent_id and not item.get('who_assembled'):
                item['who_assembled'] = agent_id
                updated = True

        return updated

    def _infer_origins_for_new_item(
        self,
        item_type: str,
        *,
        available_origin_ids: List[str],
        prev_agent_holding: Dict[str, Optional[str]],
        prev_counter_items: Dict[Tuple[int, int], Optional[str]],
        agent_id: Optional[str] = None,
        counter_key: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Infer origins for newly created transformed items during state sync."""

        def _id_type(item_id: Optional[str]) -> Optional[str]:
            if not item_id or item_id not in self.items:
                return None
            return self.items[item_id]['type']

        def _take_if_type(candidate_id: Optional[str], expected_type: str) -> Optional[str]:
            if (
                candidate_id
                and candidate_id in available_origin_ids
                and _id_type(candidate_id) == expected_type
            ):
                available_origin_ids.remove(candidate_id)
                return candidate_id
            return None

        def _take_first(expected_type: str) -> Optional[str]:
            for candidate_id in list(available_origin_ids):
                if _id_type(candidate_id) == expected_type:
                    available_origin_ids.remove(candidate_id)
                    return candidate_id
            return None

        if item_type == 'tomato_cut':
            tomato_id = None
            if counter_key is not None:
                tomato_id = _take_if_type(prev_counter_items.get(counter_key), 'tomato')
            if tomato_id is None and agent_id:
                tomato_id = _take_if_type(prev_agent_holding.get(agent_id), 'tomato')
            if tomato_id is None:
                tomato_id = _take_first('tomato')
            return {'tomato_id': tomato_id} if tomato_id else {}

        if item_type == 'tomato_salad':
            plate_id = None
            tomato_cut_id = None

            local_candidates = []
            if counter_key is not None:
                local_candidates.append(prev_counter_items.get(counter_key))
            if agent_id:
                local_candidates.append(prev_agent_holding.get(agent_id))

            for candidate_id in local_candidates:
                if plate_id is None:
                    plate_id = _take_if_type(candidate_id, 'plate')
                if tomato_cut_id is None:
                    tomato_cut_id = _take_if_type(candidate_id, 'tomato_cut')

            if plate_id is None:
                plate_id = _take_first('plate')
            if tomato_cut_id is None:
                tomato_cut_id = _take_first('tomato_cut')

            origins: Dict[str, str] = {}
            if plate_id:
                origins['plate_id'] = plate_id
            if tomato_cut_id:
                origins['tomato_cut_id'] = tomato_cut_id
            return origins

        return {}

    def sync_with_game_state(
        self,
        game: Any,
        agent_ids: List[str],
        counter_positions: List[Tuple[int, int]],
        tick: Optional[int] = None,
        second: Optional[float] = None,
    ):
        """
        Sync tracker state with actual game state to prevent desynchronization.
        
        This reads the real game state and updates internal tracking to match reality:
        - Reads what each agent is actually holding from game.gameObjects[agent_id].item
        - Reads what's on each counter from game.grid.tiles[x][y].item
        - Creates item IDs for any new items discovered
        - Updates agent_holding and counter_items dictionaries
        
        Should be called before processing each batch of actions.
        """
        prev_agent_holding = dict(self.agent_holding)
        prev_counter_items = dict(self.counter_items)

        # Read current game state
        current_agent_items: Dict[str, Optional[str]] = {}
        for agent_id in agent_ids:
            obj = getattr(game, 'gameObjects', {}).get(agent_id)
            current_agent_items[agent_id] = (getattr(obj, 'item', None) or None) if obj is not None else None

        # counter_positions are 1-indexed in logger; tracker keys are 0-indexed
        counter_keys = [(cx - 1, cy - 1) for cx, cy in counter_positions]
        current_counter_items: Dict[Tuple[int, int], Optional[str]] = {}
        for key in counter_keys:
            x, y = key
            tile = game.grid.tiles[x][y]
            current_counter_items[key] = getattr(tile, 'item', None) or None

        # Track which previous IDs are already reused this sync
        used_ids = set()
        new_agent_holding: Dict[str, Optional[str]] = {aid: None for aid in agent_ids}
        new_counter_items: Dict[Tuple[int, int], str] = {}

        def _id_type(item_id: Optional[str]) -> Optional[str]:
            if not item_id or item_id not in self.items:
                return None
            return self.items[item_id]['type']

        # 1) Preserve stable IDs when same holder/location still has same item type
        for agent_id, item_name in current_agent_items.items():
            prev_id = prev_agent_holding.get(agent_id)
            if item_name and prev_id and _id_type(prev_id) == item_name and prev_id not in used_ids:
                new_agent_holding[agent_id] = prev_id
                used_ids.add(prev_id)

        for key, item_name in current_counter_items.items():
            prev_id = prev_counter_items.get(key)
            if item_name and prev_id and _id_type(prev_id) == item_name and prev_id not in used_ids:
                new_counter_items[key] = prev_id
                used_ids.add(prev_id)

        # 2) Detect pickup continuity: counter -> agent
        emptied_counters = [
            key for key, prev_id in prev_counter_items.items()
            if prev_id and key in current_counter_items and not current_counter_items[key]
        ]
        for agent_id, item_name in current_agent_items.items():
            if not item_name or new_agent_holding[agent_id] is not None:
                continue
            matched_id = None
            for key in emptied_counters:
                cand = prev_counter_items.get(key)
                if cand and cand not in used_ids and _id_type(cand) == item_name:
                    matched_id = cand
                    break
            if matched_id:
                new_agent_holding[agent_id] = matched_id
                used_ids.add(matched_id)

        # 3) Detect drop continuity: agent -> counter
        for key, item_name in current_counter_items.items():
            if not item_name or key in new_counter_items:
                continue
            matched_id = None
            for agent_id, prev_id in prev_agent_holding.items():
                if not prev_id or prev_id in used_ids:
                    continue
                if _id_type(prev_id) != item_name:
                    continue
                # agent no longer has this item after step
                now_id = new_agent_holding.get(agent_id)
                now_name = current_agent_items.get(agent_id)
                if now_id != prev_id and now_name != item_name:
                    matched_id = prev_id
                    break
            if matched_id:
                new_counter_items[key] = matched_id
                used_ids.add(matched_id)

        # 4) Reuse any remaining unassigned previous IDs by type
        remaining_prev_ids = [iid for iid in list(prev_agent_holding.values()) + list(prev_counter_items.values()) if iid and iid not in used_ids]

        for agent_id, item_name in current_agent_items.items():
            if not item_name or new_agent_holding[agent_id] is not None:
                continue
            matched_id = None
            for cand in remaining_prev_ids:
                if _id_type(cand) == item_name and cand not in used_ids:
                    matched_id = cand
                    break
            if matched_id:
                new_agent_holding[agent_id] = matched_id
                used_ids.add(matched_id)

        for key, item_name in current_counter_items.items():
            if not item_name or key in new_counter_items:
                continue
            matched_id = None
            for cand in remaining_prev_ids:
                if _id_type(cand) == item_name and cand not in used_ids:
                    matched_id = cand
                    break
            if matched_id:
                new_counter_items[key] = matched_id
                used_ids.add(matched_id)

        consumed_origin_ids: List[str] = []
        for previous_id in list(prev_agent_holding.values()) + list(prev_counter_items.values()):
            if previous_id and previous_id not in used_ids and previous_id not in consumed_origin_ids:
                consumed_origin_ids.append(previous_id)

        # 5) Create IDs for truly new items
        for agent_id, item_name in current_agent_items.items():
            if item_name and new_agent_holding[agent_id] is None:
                inferred_origins = self._infer_origins_for_new_item(
                    item_name,
                    available_origin_ids=consumed_origin_ids,
                    prev_agent_holding=prev_agent_holding,
                    prev_counter_items=prev_counter_items,
                    agent_id=agent_id,
                )
                new_agent_holding[agent_id] = self.create_item(
                    item_name,
                    agent_id,
                    origins=inferred_origins,
                    tick=tick,
                    second=second,
                    source='game_state_sync',
                )

        for key, item_name in current_counter_items.items():
            if item_name and key not in new_counter_items:
                inferred_origins = self._infer_origins_for_new_item(
                    item_name,
                    available_origin_ids=consumed_origin_ids,
                    prev_agent_holding=prev_agent_holding,
                    prev_counter_items=prev_counter_items,
                    counter_key=key,
                )
                new_counter_items[key] = self.create_item(
                    item_name,
                    '',
                    origins=inferred_origins,
                    tick=tick,
                    second=second,
                    source='game_state_sync',
                )

        # Update touches and counters_used based on final mapping
        for agent_id, item_id in new_agent_holding.items():
            if item_id and item_id in self.items:
                if agent_id not in self.items[item_id]['touched_by']:
                    self.items[item_id]['touched_by'].append(agent_id)
                self.items[item_id]['last_touched'] = agent_id

        for key, item_id in new_counter_items.items():
            if item_id and item_id in self.items:
                prev_loc = None
                for old_key, old_id in prev_counter_items.items():
                    if old_id == item_id:
                        prev_loc = old_key
                        break
                if prev_loc != key:
                    self.items[item_id]['counters_used'].append((key, self.items[item_id].get('last_touched', ''), 'drop'))

        self.agent_holding = new_agent_holding
        self.counter_items = new_counter_items
    
    def _get_all_origins(self, item_id: str, visited: Optional[set] = None) -> List[str]:
        """Get all origin IDs recursively."""
        if item_id not in self.items:
            return []
        if visited is None:
            visited = set()
        if item_id in visited:
            return []
        visited.add(item_id)
        origins = []
        for origin_id in self.items[item_id]['origins'].values():
            if not origin_id or origin_id in visited:
                continue
            origins.append(origin_id)
            origins.extend(self._get_all_origins(origin_id, visited))
        return origins

    def _empty_data(self) -> Dict:
        """Return empty item data structure."""
        return {
            'item_id': '', 'item_type': '', 'creation_index': '', 'created_by': '',
            'created_tick': '', 'created_second': '', 'created_source': '',
            'tomato_id': '', 'plate_id': '', 'tomato_cut_id': '', 'tomato_salad_id': '', 'tomato_delivered_id': '',
            'last_touched': '', 'touched_list': '', 'touched_list_history': '',
            'is_item_collaboration': False, 'is_history_collaboration': False,
            'who_picked_tomato': '', 'who_picked_plate': '', 'who_cutted': '',
            'who_assembled': '', 'who_delivered': '', 'number_of_counters_used': 0
        }
