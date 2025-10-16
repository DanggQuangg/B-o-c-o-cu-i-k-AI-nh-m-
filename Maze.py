import pygame
import sys
from collections import deque
from queue import PriorityQueue
import math, random

# =========================
# CẤU HÌNH / THAM SỐ DỄ CHỈNH
# =========================
CELL_SIZE = 48          # Kích thước 1 ô
MARGIN = 2              # Viền giữa các ô
FONT_NAME = "consolas"
LEFTBAR_W = 400         # bảng bên trái
SIDEBAR_W = 400         # bảng bên phải: BẢNG IN TRẠNG THÁI BƯỚC ĐI
STATUS_PANEL_H = 160    # chiều cao vùng in trạng thái ở bảng trái
# Đánh dấu ô đã đi
COLOR_VISITED = (255, 235, 120)
VISITED_DOT_RADIUS = 6

# Chi phí di chuyển theo độ dốc
base_move_cost = 1.0
cost_per_height = 0.2
max_slope = 3

# Màu sắc
COLOR_BG = (18, 18, 22)
COLOR_GRID = (40, 40, 48)
COLOR_HOLE = (0, 0, 0)
COLOR_FLAT = (70, 120, 70)
COLOR_DOWN_MILD = (80, 140, 200)
COLOR_DOWN_STEEP = (120, 180, 240)
COLOR_UP_MILD = (200, 160, 60)
COLOR_UP_STEEP = (220, 100, 60)
COLOR_TOO_STEEP = (120, 0, 0)
COLOR_START = (180, 255, 180)
COLOR_GOAL = (255, 220, 120)
COLOR_TEXT = (230, 230, 230)
COLOR_WARNING = (255, 120, 120)
COLOR_BTN = (60, 62, 70)
COLOR_BTN_HOVER = (85, 88, 100)
COLOR_BTN_TEXT = (240, 240, 245)
COLOR_PANEL = (30, 32, 38)
COLOR_SEP = (55, 58, 66)

# =========================
# MAP
# =========================
RAW_MAP = [
    [ "S",  5, "X",  3,  2, "X", 2, 0, 0, "X",  4,  3,  3, "X",  1,  2,  5,  1,  3, "X"],
    [ 1,  2, "X",  4,  2,  4,  2,  4,  5, "X",  5,  1, "X",  4,  4,  1,  3,  2, 3,  3],
    [ 3,  4, "X",  3, "X", "X", "X",  3,  1, "X",  1,  1,  3, "X",  2, "X",  5,  2, "X",  3],
    [ 4,  1,  1,  1, "X",  3,  1,  4,  2, "X",  5,  2,  1, "X",  2,  5,  3,  1, "X",  1],
    [ 5,  4,  2,  1, "X",  2,  5, "X", "X", "X", "X", "X",  1, "X", "X", "X",  3,  5, "X",  2],
    [ 5,  4,  1, "X", "X",  3,  3, "X",  3,  2,  4,  5,  2,  5,  2,  1,  1,  4, "X", "G"],
    [ 2,  5,  4, "X",  5,  2,  4, "X",  5,  2,  5, "X",  2, "X",  2,  3,  2,  5, "X",  2],
    [ 4,  1, "X",  3,  5,  3,  2,  2,  3,  5,  5, "X",  2,  3,  4, "X", "X", "X", "X",  5],
    ["X",  1, "X",  4, "X", "X", "X", "X", "X", "X",  1, "X",  2,  1,  2, "X",  2,  3,  5,  5],
    [ 1,  3,  5, "X",  3,  5,  3,  4,  "X",  1,  4, "X",  2,  3, "X",  2,  5,  1, "X",  3],
    [ 5,  3,  3, "X",  2,  1,  0,  5,  "X",  1,  3, "X", "X",  5, "X",  2,  4,  4, "X",  2],
    [ 4,  4,  5, "X", "X", "X", "X",  3,  1,  1,  2,  5,  2,  2, "X",  3,  3,  5, "X",  5],
    [ 1, "X",  3,  2,  2,  2,  0,  4,  1, "X",  3,  1,  1, "X",  4,  2, "X",  2, "X",  4],
    [ 2,  5,  2, "X",  1,  5,  0,  1,  1, "X",  4,  2,  2,  2,  5,  4,  5,  5, "X", 2],
    [ 3,  3,  5, "X",  3,  2,  2, "X",  4, "X",  5,  5,  2,  4,  3,  5,  5,  "X",  1,  2],
    [ 4,  0,  4, "X",  1,  2,  2,  4,  1, "X", "X", "X", "X", "X", "X",  2,  5,  3,  4,  4],
    [ 1,  2, "X",  5,  5,  1,  3, "X",  4,  2,  1, "X",  2,  3,  2,  4,  5,  1,  4,  1],
    [ 3,  0,  2,  2,  3,  4, 2,  5, "X",  2,  1,  3,  1, "X",  4,  5,  2, "X",  3,  5],
    [ 3,  "X",  2, "X",  "X",  2,  "X",  "X",  0,  "X", "X",  3,  4, "X",  2,  3,  "X",  "X",  1,  1],
    [ 2,  2,  4, 3, 5, 4, 3, 2, 1, 2, "X",  3,  1, "X",  5,  4,  4,  "X",  2,  5],
]
ROWS = len(RAW_MAP)
COLS = len(RAW_MAP[0])

# =========================
# HÀM TIỆN ÍCH
# =========================
def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def neighbors4(r, c):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        rr, cc = r+dr, c+dc
        if in_bounds(rr,cc):
            yield rr, cc

def legal_neighbors(grid, r, c):
    for rr, cc in neighbors4(r, c):
        if grid[rr][cc]["hole"]:
            continue
        cst = move_cost(grid[r][c]["h"], grid[rr][cc]["h"])
        if cst is not None:
            yield (rr, cc), cst

def move_cost(cur_h, next_h):
    dh = next_h - cur_h
    if abs(dh) > max_slope:
        return None
    return base_move_cost + dh * cost_per_height

def find_start_goal(raw_map):
    start = None
    goal = None
    grid = [[{"hole": False, "h": 0, "start": False, "goal": False} for _ in range(COLS)] for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            v = raw_map[r][c]
            if v == "X":
                grid[r][c]["hole"] = True
                grid[r][c]["h"] = 0
            elif v == "S":
                start = (r, c)
                grid[r][c]["start"] = True
                grid[r][c]["h"] = 0
            elif v == "G":
                goal = (r, c)
                grid[r][c]["goal"] = True
                neigh = []
                for rr, cc in neighbors4(r,c):
                    vv = raw_map[rr][cc]
                    if isinstance(vv, int):
                        neigh.append(vv)
                grid[r][c]["h"] = round(sum(neigh)/len(neigh)) if neigh else 0
            else:
                grid[r][c]["h"] = int(v)
    return grid, start, goal

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _mix_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * _clamp(t,0,1)) for i in range(3))

def draw_fuel_gauge(surface, rect, fuel, start_fuel, font, title="NHIÊN LIỆU"):
    pygame.draw.rect(surface, COLOR_PANEL, rect, border_radius=12)
    pygame.draw.rect(surface, COLOR_SEP, rect, 2, border_radius=12)

    PAD = 12
    inner = rect.inflate(-2*PAD, -2*PAD)

    title_surf = font.render(title, True, COLOR_TEXT)
    surface.blit(title_surf, (inner.centerx - title_surf.get_width()//2, inner.y))

    if start_fuel <= 0:
        fuel_clamped = 0.0
        pct = 0.0
    else:
        fuel_clamped = _clamp(fuel, 0.0, start_fuel)
        pct = _clamp(fuel_clamped / start_fuel, 0.0, 1.0)
    pct_int = int(round(pct * 100))

    GAP_T = 10
    content_top = inner.y + title_surf.get_height() + GAP_T
    content_h = max(0, inner.bottom - content_top)

    BAR_MIN_W, BAR_MAX_W = 26, 40
    bar_w = int(_clamp(int(inner.w * 0.28), BAR_MIN_W, BAR_MAX_W))
    COL_GAP = 16

    if content_h < 40 or inner.w < (bar_w + COL_GAP + 80):
        summary = f"{fuel_clamped:.2f}/{start_fuel:.2f}  •  {pct_int}%"
        sum_surf = font.render(summary, True, COLOR_TEXT if pct >= 0.15 else COLOR_WARNING)
        surface.blit(sum_surf, (inner.x, content_top))
        return

    bar_rect = pygame.Rect(inner.x, content_top, bar_w, content_h)
    info_x = bar_rect.right + COL_GAP
    info_rect = pygame.Rect(info_x, content_top, max(0, inner.right - info_x), content_h)

    pygame.draw.rect(surface, (24,24,28), bar_rect, border_radius=8)
    pygame.draw.rect(surface, COLOR_SEP, bar_rect, 2, border_radius=8)

    if pct >= 0.5:
        fill_col = _mix_color((255, 220, 0), (60, 200, 120), (pct - 0.5) / 0.5)
    else:
        fill_col = _mix_color((220, 60, 60), (255, 220, 0), pct / 0.5)

    inner_bar = bar_rect.inflate(-6, -6)
    fill_h = int(inner_bar.h * pct)
    fill_rect = pygame.Rect(inner_bar.x, inner_bar.bottom - fill_h, inner_bar.w, max(0, fill_h))

    if pct < 0.15 and (pygame.time.get_ticks() // 300) % 2 == 0:
        fill_col = _mix_color(fill_col, (255, 255, 255), 0.35)

    if fill_h > 0:
        pygame.draw.rect(surface, fill_col, fill_rect, border_radius=6)

    tick_labels = [100, 50, 0] if inner_bar.h >= 120 else [100, 0]
    for label in tick_labels:
        yy = inner_bar.y + int((inner_bar.h) * (1 - label/100.0))
        pygame.draw.line(surface, (80,80,88), (inner_bar.x, yy), (inner_bar.right, yy), 1)
        if inner_bar.w >= 34:
            lab = font.render(str(label), True, (160,160,168))
            surface.blit(lab, (inner_bar.centerx - lab.get_width()//2, yy - lab.get_height()//2))

    line_y = info_rect.y + 2
    txt1 = font.render(f"{fuel_clamped:.2f} / {start_fuel:.2f}", True, COLOR_TEXT)
    surface.blit(txt1, (info_rect.x, line_y)); line_y += txt1.get_height() + 4

    pct_color = COLOR_TEXT if pct >= 0.15 else COLOR_WARNING
    txt2 = font.render(f"{pct_int}%", True, pct_color)
    surface.blit(txt2, (info_rect.x, line_y)); line_y += txt2.get_height() + 8

    remain_h = info_rect.bottom - line_y
    BAR_H = 12
    if remain_h >= BAR_H + 4:
        pct_bg = pygame.Rect(info_rect.x, line_y, max(60, info_rect.w), BAR_H)
        pygame.draw.rect(surface, (24,24,28), pct_bg, border_radius=6)
        pygame.draw.rect(surface, COLOR_SEP, pct_bg, 1, border_radius=6)
        pct_fill = pct_bg.inflate(-2, -2)
        pct_fill.width = int(max(0, pct_fill.w * pct))
        if pct_fill.width > 0:
            pygame.draw.rect(surface, fill_col, pct_fill, border_radius=5)

# =========================
# CLASS XE
# =========================
class Car:
    FILENAMES = {
        (-1, 0): "car_up.png",
        ( 0, 1): "car_right.png",
        ( 1, 0): "car_down.png",
        ( 0,-1): "car_left.png",
    }
    def __init__(self, start_rc, cell_px):
        self.r, self.c = start_rc
        self.cell_px = cell_px
        self.cur_dir = (0, 1)
        self.images = {}
        self._load_images()
    def _load_images(self):
        self.images = {}
        for d, fname in Car.FILENAMES.items():
            try:
                img = pygame.image.load(fname).convert_alpha()
                self.images[d] = img
            except Exception:
                self.images[d] = None
    def _set_dir_by_delta(self, dr, dc):
        if (dr, dc) in Car.FILENAMES:
            self.cur_dir = (dr, dc)
    def try_move(self, dr, dc, grid, fuel):
        if dr == 0 and dc == 0:
            return False, fuel, "Không có hướng di chuyển."
        nr, nc = self.r + dr, self.c + dc
        if not in_bounds(nr, nc):
            self._set_dir_by_delta(dr, dc)
            return False, fuel, "Không thể đi: ngoài bản đồ."
        if grid[nr][nc]["hole"]:
            self._set_dir_by_delta(dr, dc)
            return False, fuel, "Không thể đi: ô hố (X)."
        cur_h = grid[self.r][self.c]["h"]
        nxt_h = grid[nr][nc]["h"]
        cst = move_cost(cur_h, nxt_h)
        if cst is None:
            self._set_dir_by_delta(dr, dc)
            return False, fuel, f"Quá dốc (|Δh|>{max_slope})."
        if cst > fuel + 1e-9:
            self._set_dir_by_delta(dr, dc)
            return False, fuel, f"Không đủ xăng (cần {cst:.1f})."
        new_fuel = max(0.0, fuel - cst)
        self.r, self.c = nr, nc
        self._set_dir_by_delta(dr, dc)
        return True, new_fuel, f"Đi (-{cst:.1f} xăng)."
    def draw(self, surface, rect):
        img = self.images.get(self.cur_dir)
        if img is None:
            pygame.draw.rect(surface, (250, 250, 255), rect, border_radius=10)
            pygame.draw.rect(surface, (50, 50, 60), rect, 2, border_radius=10)
            return
        scaled = pygame.transform.smoothscale(img, (rect.width, rect.height))
        surface.blit(scaled, rect.topleft)

# =========================
# TILE ẢNH THEO ĐỘ CAO + HỐ
# =========================
class HeightTiles:
    def __init__(self, cell_px, margin):
        self.cell_px = cell_px
        self.margin = margin
        self._cache_by_rc = {}
        self._raw_heights = {}
        self._raw_misc = {}
    def load_height_tile(self, h, filename):
        img = pygame.image.load(filename).convert_alpha()
        self._raw_heights[h] = img
    def load_misc(self, key, filename):
        img = pygame.image.load(filename).convert_alpha()
        self._raw_misc[key] = img
    def _scaled(self, surf, inner_w, inner_h):
        key = (id(surf), inner_w, inner_h)
        cache = self._cache_by_rc.get(key)
        if cache is None:
            cache = pygame.transform.smoothscale(surf, (inner_w, inner_h))
            self._cache_by_rc[key] = cache
        return cache
    def get_for_height(self, h, inner_rect):
        surf = self._raw_heights.get(h)
        if surf is None:
            if self._raw_heights:
                nearest_h = min(self._raw_heights.keys(), key=lambda k: abs(k - h))
                surf = self._raw_heights[nearest_h]
            else:
                return None
        return self._scaled(surf, inner_rect.w, inner_rect.h)
    def get_misc(self, key, inner_rect):
        surf = self._raw_misc.get(key)
        if surf is None:
            return None
        return self._scaled(surf, inner_rect.w, inner_rect.h)

# =========================
# AC-3 STYLE REDUCTION
# =========================
def ac3_reduce_map(grid, start, goal):
    def ok_cell(r, c):
        return in_bounds(r, c) and (not grid[r][c]["hole"])
    adj = { (r, c): [] for r in range(ROWS) for c in range(COLS) if ok_cell(r, c) }
    for (r, c) in list(adj.keys()):
        ch = grid[r][c]["h"]
        for rr, cc in neighbors4(r, c):
            if ok_cell(rr, cc) and abs(grid[rr][cc]["h"] - ch) <= max_slope:
                adj[(r, c)].append((rr, cc))
    F = set(); q = deque()
    if ok_cell(*start):
        F.add(start); q.append(start)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in F:
                F.add(v); q.append(v)
    B = set(); rq = deque()
    if ok_cell(*goal):
        B.add(goal); rq.append(goal)
    inv = {k: [] for k in adj}
    for u, vs in adj.items():
        for v in vs:
            inv[v].append(u)
    while rq:
        v = rq.popleft()
        for u in inv.get(v, []):
            if u not in B:
                B.add(u); rq.append(u)
    keep = F & B
    removed = 0
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == start or (r, c) == goal:
                continue
            if not grid[r][c]["hole"] and (r, c) not in keep:
                grid[r][c]["hole"] = True
                removed += 1
    return removed, keep

# =========================
# IDS
# =========================
def ids_path(grid, start, goal, fuel_budget, max_depth=None):
    if max_depth is None:
        max_depth = ROWS * COLS
    def dls(node, depth, limit, used_cost, path, visited):
        if node == goal:
            return path[:]
        if depth == limit:
            return None
        r, c = node
        for (rr, cc), step_cost_val in legal_neighbors(grid, r, c):
            nxt = (rr, cc)
            if nxt in visited:
                continue
            new_cost = used_cost + step_cost_val
            if new_cost > fuel_budget + 1e-9:
                continue
            visited.add(nxt)
            path.append(nxt)
            res = dls(nxt, depth+1, limit, new_cost, path, visited)
            if res is not None:
                return res
            path.pop()
        return None
    for limit in range(0, max_depth + 1):
        res = dls(start, 0, limit, 0.0, [start], {start})
        if res is not None:
            return res
    return None

# =========================
# A* (heuristic có độ cao)
# =========================
def a_star_path(grid, start, goal, fuel_budget):
    alpha = 0.2
    def h(rc):
        r, c = rc
        return abs(r - goal[0]) + abs(c - goal[1]) + (alpha *(grid[rr][cc]["h"]) - grid[goal[0]][goal[1]]["h"])

    pq = PriorityQueue()
    pq.put((0.0, start))              
    g_score = {start: 0.0}            
    came_from = {}                    
    closed = set()                    

    while not pq.empty():
        f, cur = pq.get()             
        if cur in closed:
            continue
        closed.add(cur)

        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        r, c = cur
        for rr, cc in neighbors4(r, c):       
            nxt = (rr, cc)
            if grid[rr][cc]["hole"]:
                continue
            cst = move_cost(grid[r][c]["h"], grid[rr][cc]["h"])
            if cst is None:                 
                continue
            ng = g_score[cur] + cst           
            if ng > fuel_budget + 1e-9:
                continue
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = cur
                pq.put((ng + h(nxt), nxt))    
    return None

# =========================
# BFS
# =========================
def bfs_path(grid, start, goal, fuel_budget):
    q = deque([start])
    parent = {start: None}
    used_cost = {start: 0.0}
    while q:
        u = q.popleft()
        if u == goal:
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        r, c = u
        for (v, cst) in legal_neighbors(grid, r, c):
            new_cost = used_cost[u] + cst
            if new_cost > fuel_budget + 1e-9:
                continue
            if v not in parent:
                parent[v] = u
                used_cost[v] = new_cost
                q.append(v)
    return None

# =========================
# UCS
# =========================
def ucs_path(grid, start, goal, fuel_budget):
    pq = PriorityQueue()
    pq.put((0.0, start))
    cost = {start: 0.0}
    parent = {}
    while not pq.empty():
        g, u = pq.get()
        if u == goal:
            path = []
            cur = u
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return path
        r, c = u
        for (v, cst) in legal_neighbors(grid, r, c):
            ng = g + cst
            if ng > fuel_budget + 1e-9:
                continue
            if v not in cost or ng < cost[v]:
                cost[v] = ng
                parent[v] = u
                pq.put((ng, v))
    return None

# =========================
# Simulated Annealing
# =========================
def gen_moves(grid, state, goal=None):
    r, c = state
    cur_h = grid[r][c]["h"]
    for rr, cc in neighbors4(r, c):
        if grid[rr][cc]["hole"]:
            continue
        if move_cost(cur_h, grid[rr][cc]["h"]) is not None:
            yield (rr, cc)
def can_follow_path_with_fuel(grid, path, fuel):
    if not path or len(path) < 2:
        return True  # không (hoặc gần như không) di chuyển => không tốn xăng


    remaining = fuel
    for i in range(len(path) - 1):
        r, c = path[i]
        nr, nc = path[i + 1]


        # bước tới ô hố thì không hợp lệ
        if grid[nr][nc]["hole"]:
            return False


        cur_h = grid[r][c]["h"]
        nxt_h = grid[nr][nc]["h"]


        cst = move_cost(cur_h, nxt_h)
        # quá dốc hoặc không đủ xăng
        if cst is None or cst > remaining + 1e-9:
            return False


        remaining -= cst


    return True

def sa_score(grid, s, goal, w_height=0.2):
    (r, c), (gr, gc) = s, goal
    manhattan = abs(r - gr) + abs(c - gc)
    dh = grid[r][c]["h"] - grid[gr][gc]["h"]
    return manhattan + w_height * max(0, dh)

def sa_path(grid, start, goal,
            fuel_budget=None,          # mới
            T=20.0, alpha=0.995,
            K=200,
            max_iters=200_000,
            w_height=0.2,
            seed=None,
            on_fail=None):             # mới


    if seed is not None:
        random.seed(seed)


    if start == goal:
        # Ở ngay đích: không tốn xăng
        return [start], 1


    trail = [start]
    cur = start
    cur_score = sa_score(grid, cur, goal, w_height=w_height)


    visited_count = 0
    iters = 0


    while iters < max_iters and T > 1e-6:
        for _ in range(K):
            iters += 1


            # Nếu đã tới goal => CHỈ trả về nếu đủ xăng
            if cur == goal:
                if fuel_budget is None or can_follow_path_with_fuel(grid, trail, fuel_budget):
                    return trail
                # Không tìm tiếp, chỉ báo và trả None
                if on_fail:
                    on_fail("không thể tìm được đường đi với lượng nhiên liệu hiện tại")
                return None


            neighs = list(gen_moves(grid, cur))
            visited_count += 1


            if not neighs:
                # lùi lại 1 bước nếu có, nếu không thì ở nguyên start
                if len(trail) > 1:
                    trail.pop()
                    cur = trail[-1]
                    cur_score = sa_score(grid, cur, goal, w_height=w_height)
                else:
                    cur = start
                    trail = [start]
                    cur_score = sa_score(grid, cur, goal, w_height=w_height)
                continue


            nxt = random.choice(neighs)
            nxt_score = sa_score(grid, nxt, goal, w_height=w_height)
            delta = nxt_score - cur_score  # <0 là tốt hơn


            accept = (delta <= 0) or (random.random() < math.exp(-delta / max(T, 1e-12)))


            if accept:
                if nxt in trail:
                    idx = trail.index(nxt)
                    trail = trail[:idx+1]
                else:
                    trail.append(nxt)


                cur = nxt
                cur_score = nxt_score


                if cur == goal:
                    if fuel_budget is None or can_follow_path_with_fuel(grid, trail, fuel_budget):
                        return trail
                    if on_fail:
                        on_fail("không thể tìm được đường đi với lượng nhiên liệu hiện tại")
                    return None


        T *= alpha


    # Hết vòng “ủ” mà chưa tới đích
    return None


# =========================
# Uncertain Action (Value Iteration)
# =========================
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def _dir_to_delta(action):
    if action == "UP":    return (-1, 0)
    if action == "DOWN":  return ( 1, 0)
    if action == "LEFT":  return ( 0,-1)
    if action == "RIGHT": return ( 0, 1)
    raise ValueError("Unknown action")


def delta_to_dir(dr, dc): 
    if   (dr, dc) == (-1, 0): return "UP"
    elif (dr, dc) == ( 1, 0): return "DOWN"
    elif (dr, dc) == ( 0,-1): return "LEFT"
    elif (dr, dc) == ( 0, 1): return "RIGHT"
    return None


def _left_of(action): 
    m = {"UP":"LEFT", "LEFT":"DOWN", "DOWN":"RIGHT", "RIGHT":"UP"}
    return m[action]


def _try_step(grid, s, action):
    r, c = s
    dr, dc = _dir_to_delta(action)
    rr, cc = r + dr, c + dc
    if not in_bounds(rr, cc):                # ngoài bản đồ
        return None
    if grid[rr][cc]["hole"]:                 # ô hố
        return None
    if move_cost(grid[r][c]["h"], grid[rr][cc]["h"]) is None:  # quá dốc
        return None
    return (rr, cc)


def uncertain_results(grid, s, action):
    outcomes = set()


    s_int = _try_step(grid, s, action)
    if s_int is not None:
        outcomes.add(s_int)


    s_left = _try_step(grid, s, _left_of(action))
    if s_left is not None:
        outcomes.add(s_left)
    return outcomes


def actions_ordered_toward(s, goal):
    """Ưu tiên các hướng tiến gần goal trước để giảm nổ nhánh."""
    gr, gc = goal; r, c = s
    order = []
    if gr < r: order.append("UP")
    if gr > r: order.append("DOWN")
    if gc < c: order.append("LEFT")
    if gc > c: order.append("RIGHT")
    for a in ["UP", "DOWN", "LEFT", "RIGHT"]:
        if a not in order:
            order.append(a)
    return order
def and_or_graph_search(grid, start, goal, depth_limit=600): 
    memo = {}
    stack = set()
    return _or_search(grid, start, goal, stack, memo, depth_limit)


def _or_search(grid, s, goal, stack, memo, depth_left):
    if is_goal(s, goal):
        memo[s] = "SUCCESS"; return "SUCCESS"
    if depth_left <= 0: return None 
    if s in stack: return None
    if s in memo:
        return memo[s] if memo[s] != "FAIL" else None


    stack.add(s) 
    best_plan = None
    for action in actions_ordered_toward(s, goal):
       
        result_states = uncertain_results(grid, s, action)
        if not result_states: 
            continue


        subplan = _and_search(grid, result_states, goal, stack, memo, depth_left - 1)
       
        if subplan is not None:
            best_plan = ("DO", action, subplan) 
            break
    stack.remove(s)


    memo[s] = best_plan if best_plan is not None else "FAIL"
    return best_plan


def _and_search(grid, result_states, goal, stack, memo, depth_left): 
    subplans = {}


    for sp in result_states:
        #gọi or search
        plan_p = _or_search(grid, sp, goal, stack, memo, depth_left)
 
        if plan_p is None:
            return None
        subplans[sp] = plan_p
    return subplans


def next_action_from_plan(plan, state):
    if plan is None:
        return None
    if plan == "SUCCESS":
        return None
    tag, action, _ = plan
    assert tag == "DO"
    return action


def advance_plan_after_outcome(plan, outcome_state): 
    if plan is None or plan == "SUCCESS":
        return plan
    tag, action, subplans = plan
    assert tag == "DO"
    return subplans.get(outcome_state, None)


def stochastic_step(grid, s, action):
    import random
    outs = list(uncertain_results(grid, s, action))
    if not outs:
        return s 
    return random.choice(outs)


def is_goal(state, goal):
    return state == goal



# =========================
# Hill Climbing
# =========================
def hill_climbing_path(grid, start, goal):
    def h(rc):
        r, c = rc
        return abs(r - goal[0]) + abs(c - goal[1])
    cur = start
    path = [cur]
    closed = set([cur])
    while True:
        if cur == goal:
            return path
        r, c = cur
        best = None
        best_h = h(cur)
        for rr, cc in neighbors4(r, c):
            if grid[rr][cc]["hole"]:
                continue
            if move_cost(grid[r][c]["h"], grid[rr][cc]["h"]) is None:
                continue
            nxt_h = h((rr, cc))
            if nxt_h < best_h and (rr, cc) not in closed:
                best_h = nxt_h 
                best = (rr, cc)
        if best is None or cur == goal:
            return path
        cur = best
        closed.add(cur)
        path.append(cur)

# =========================
# Forward Checking + Backtracking không đệ quy (giới hạn độ sâu)
# =========================
def _manhattan(a, b):
    (r1, c1), (r2, c2) = a, b
    return abs(r1 - r2) + abs(c1 - c2)

def _ordered_neighbors_for_fc(grid, state, goal):
    r, c = state
    cur_h = grid[r][c]["h"]
    neighs = []
    for rr, cc in neighbors4(r, c):
        if grid[rr][cc]["hole"]:
            continue
        nxt_h = grid[rr][cc]["h"]
        if move_cost(cur_h, nxt_h) is None:  # quá dốc
            continue
        neighs.append((rr, cc))
    neighs.sort(key=lambda rc: _manhattan(rc, goal))
    return neighs

def _has_reachable_path_lookahead(grid, s, goal, blocked):
    if s == goal:
        return True

    q = deque([s])
    seen = {s}
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        r, c = cur
        cur_h = grid[r][c]["h"]
        for rr, cc in neighbors4(r, c):
            nxt = (rr, cc)
            if nxt in seen or nxt in blocked:
                continue
            if grid[rr][cc]["hole"]:
                continue
            nxt_h = grid[rr][cc]["h"]
            if move_cost(cur_h, nxt_h) is None:
                continue
            seen.add(nxt)
            q.append(nxt)
    return False

def forward_checking_path(grid, start, goal, depth_limit=None, fuel_budget=None, on_fail=None):
    if start == goal:
        return [start]
    if depth_limit is None:
        depth_limit = ROWS * COLS
    path = [start]
    used = {start}
    visited_count = 0
    def make_frame(state, depth):
        return {
            "state": state,
            "depth": depth,
            "neighs": _ordered_neighbors_for_fc(grid, state, goal),
            "idx": 0,
            "expanded": False
        }
    stack = [make_frame(start, depth_limit)]
    while stack:
        fr = stack[-1]
        cur = fr["state"]


        # Chạm goal -> chỉ trả về nếu đủ xăng
        if cur == goal:
            if fuel_budget is None or can_follow_path_with_fuel(grid, path, fuel_budget):
                return path
            if on_fail:
                on_fail("không thể tìm được đường đi với lượng nhiên liệu hiện tại")
            return None  # không tìm đường khác


        if fr["depth"] <= 0:
            stack.pop()
            if path:
                last = path.pop()
                if last in used:
                    used.remove(last)
            continue


        if not fr["expanded"]:
            visited_count += 1
            fr["expanded"] = True


        advanced = False
        while fr["idx"] < len(fr["neighs"]):
            nxt = fr["neighs"][fr["idx"]]
            fr["idx"] += 1


            if nxt in used:
                continue


            if not _has_reachable_path_lookahead(grid, nxt, goal, blocked=used | {nxt}):
                continue


            used.add(nxt)
            path.append(nxt)
            stack.append(make_frame(nxt, fr["depth"] - 1))
            advanced = True
            break
        if advanced:
            continue
        stack.pop()
        if path:
            last = path.pop()
            if last in used:
                used.remove(last)


    return None


# =========================
# Belief Conformant
# =========================
def belief_conformant_path(
    grid, start, goal, fuel_budget,
    init_radius=1,
    strict_goal_all=True,
    max_beliefs=80_000
):
    """
    Conformant planning với mô hình xác định (không sensing):
    - Belief = frozenset các trạng thái có thể đang ở.
    - Hành động hợp lệ nếu *mọi* trạng thái trong belief thực thi được hành động đó.
    - Chi phí bước = max chi phí trong các chuyển tiếp (worst-case).
    - Dijkstra/UCS trên không gian belief, ràng buộc tổng chi phí <= fuel_budget.
    - Điều kiện dừng:
        + strict_goal_all=True: mọi trạng thái trong belief đều là goal.
        + strict_goal_all=False: belief chỉ chứa đúng {goal}.
    - Trả về đường đi cụ thể cho xe (áp chuỗi action lên vị trí thật).
    """

    actions = [(-1,0),(1,0),(0,-1),(0,1)]

    # ---- Tập ô hợp lệ (không hố)
    def ok_cell(rc):
        r, c = rc
        return in_bounds(r, c) and not grid[r][c]["hole"]

    # ---- Tạo belief khởi tạo quanh start (nếu radius=0 thì chỉ có start)
    sr, sc = start
    init_cells = []
    for dr in range(-init_radius, init_radius+1):
        for dc in range(-init_radius, init_radius+1):
            rc = (sr+dr, sc+dc)
            if ok_cell(rc):
                init_cells.append(rc)
    if not init_cells:
        init_cells = [start]
    B0 = frozenset(init_cells)

    # ---- Áp dụng 1 hành động a lên belief B (nếu có state không đi được -> None)
    def apply_action(B, a):
        dr, dc = a
        next_states = []
        step_costs = []
        for (r, c) in B:
            rr, cc = r + dr, c + dc
            if not in_bounds(rr, cc) or grid[rr][cc]["hole"]:
                return None
            cst = move_cost(grid[r][c]["h"], grid[rr][cc]["h"])
            if cst is None:
                return None
            next_states.append((rr, cc))
            step_costs.append(cst)
        return frozenset(next_states), (max(step_costs) if step_costs else 0.0)

    # ---- Điều kiện goal cho belief
    def is_belief_goal(B):
        if strict_goal_all:
            return all(rc == goal for rc in B)
        # Lỏng hơn: tất cả trùng nhau và chính là goal
        return (len(B) == 1 and next(iter(B)) == goal)

    # ---- Dijkstra/UCS trên belief
    from queue import PriorityQueue
    pq = PriorityQueue()
    pq.put((0.0, B0))
    g_cost = {B0: 0.0}
    parent = {B0: None}
    parent_action = {B0: None}

    expanded = 0
    while not pq.empty():
        g, B = pq.get()
        if g > g_cost.get(B, float('inf')) + 1e-12:
            continue

        # Giới hạn phòng cháy
        expanded += 1
        if expanded > max_beliefs:
            break

        if is_belief_goal(B):
            # khôi phục chuỗi action
            actions_seq = []
            curB = B
            while parent[curB] is not None:
                actions_seq.append(parent_action[curB])
                curB = parent[curB]
            actions_seq.reverse()

            # Áp dụng lên vị trí thực tế để tạo đường đi cụ thể + kiểm tra xăng
            path = [start]
            fuel_used = 0.0
            r, c = start
            for (dr, dc) in actions_seq:
                rr, cc = r + dr, c + dc
                if not in_bounds(rr, cc) or grid[rr][cc]["hole"]:
                    return None
                cst = move_cost(grid[r][c]["h"], grid[rr][cc]["h"])
                if cst is None or fuel_used + cst > fuel_budget + 1e-9:
                    return None
                fuel_used += cst
                r, c = rr, cc
                path.append((r, c))
            return path if path and path[-1] == goal else None

        # Mở rộng 4 hướng
        for a in actions:
            res = apply_action(B, a)
            if res is None:
                continue
            B2, step_worst = res
            ng = g + step_worst
            if ng > fuel_budget + 1e-9:
                continue
            # ràng buộc giảm kích thước belief (tự nhiên do map hẹp) -> ưu tiên
            if B2 not in g_cost or ng < g_cost[B2] - 1e-12:
                g_cost[B2] = ng
                parent[B2] = B
                parent_action[B2] = a
                pq.put((ng, B2))

    return None

# =========================
# 5 THUẬT TOÁN BỔ SUNG
# =========================
def dfs_path(grid, start, goal, fuel_budget):
    stack = [(start, 0.0, [start], {start})]
    while stack:
        u, used_cost, path, visited = stack.pop()
        if u == goal:
            return path
        r, c = u
        neighs = []
        for (v, cst) in legal_neighbors(grid, r, c):
            ng = used_cost + cst
            if ng <= fuel_budget + 1e-9 and v not in visited:
                vr, vc = v
                h = abs(vr - goal[0]) + abs(vc - goal[1])
                neighs.append((h, v, cst))
        neighs.sort(key=lambda x: x[0], reverse=True)  # sâu trước
        for _h, v, cst in neighs:
            stack.append((v, used_cost + cst, path + [v], visited | {v}))
    return None

def greedy_best_first_path(grid, start, goal, fuel_budget, alpha=0.15):
    def h(rc):
        r, c = rc
        return abs(r - goal[0]) + abs(c - goal[1]) + alpha * max(0, grid[r][c]["h"] - grid[goal[0]][goal[1]]["h"])
    pq = PriorityQueue()
    pq.put((h(start), start))
    parent = {start: None}
    used_cost = {start: 0.0}
    while not pq.empty():
        _hf, u = pq.get()
        if u == goal:
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        r, c = u
        for (v, cst) in legal_neighbors(grid, r, c):
            ng = used_cost[u] + cst
            if ng > fuel_budget + 1e-9:
                continue
            if v not in used_cost or ng < used_cost[v]:
                used_cost[v] = ng
                parent[v] = u
                pq.put((h(v), v))
    return None

def beam_search_path(grid, start, goal, fuel_budget, beam_width=5, alpha=0.15, max_iters=20000):
    def h(rc):
        r, c = rc
        return abs(r - goal[0]) + abs(c - goal[1]) + alpha * max(0, grid[r][c]["h"] - grid[goal[0]][goal[1]]["h"])
    frontier = [(start, 0.0, None)]
    parents = {start: None}
    costs = {start: 0.0}
    it = 0
    while frontier and it < max_iters:
        it += 1
        for node, _g, _p in frontier:
            if node == goal:
                path = []
                cur = node
                while cur is not None:
                    path.append(cur)
                    cur = parents[cur]
                return list(reversed(path))
        candidates = []; seen_in_round = set()
        for u, g, _p in frontier:
            r, c = u
            for (v, cst) in legal_neighbors(grid, r, c):
                ng = g + cst
                if ng > fuel_budget + 1e-9:
                    continue
                if (v not in costs) or (ng < costs[v] - 1e-12):
                    costs[v] = ng; parents[v] = u
                if v not in seen_in_round:
                    seen_in_round.add(v)
                    candidates.append((h(v), v, costs[v]))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[2]))
        candidates = candidates[:beam_width]
        frontier = [(v, costs[v], parents[v]) for (_h, v, _g) in candidates]
    for node, _g, _p in frontier:
        if node == goal:
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            return list(reversed(path))
    return None

def backtracking_path(grid, start, goal, fuel_budget, depth_limit=None):
    if depth_limit is None:
        depth_limit = ROWS * COLS
    best_path = None
    seen = set()
    def dfs(u, used, depth, path):
        nonlocal best_path
        if u == goal:
            best_path = path[:]; return True
        if depth >= depth_limit:
            return False
        r, c = u
        neighs = []
        for (v, cst) in legal_neighbors(grid, r, c):
            vr, vc = v
            h = abs(vr - goal[0]) + abs(vc - goal[1])
            neighs.append((h, v, cst))
        neighs.sort(key=lambda x: x[0])
        for _h, v, cst in neighs:
            if v in seen: continue
            ng = used + cst
            if ng > fuel_budget + 1e-9: continue
            seen.add(v); path.append(v)
            if dfs(v, ng, depth + 1, path): return True
            path.pop(); seen.remove(v)
        return False
    seen.add(start)
    dfs(start, 0.0, 0, [start])
    return best_path

def belief_partial_path(grid, start, goal, fuel_budget, init_radius=1, min_feasible_ratio=0.6, use_avg_cost=True):
    actions = [(-1,0),(1,0),(0,-1),(0,1)]
    sr, sc = start
    init = []
    for dr in range(-init_radius, init_radius+1):
        for dc in range(-init_radius, init_radius+1):
            r, c = sr + dr, sc + dc
            if in_bounds(r, c) and (not grid[r][c]["hole"]):
                init.append((r, c))
    if not init: init = [start]
    B0 = frozenset(init)
    def apply_action_partial(belief, a):
        dr, dc = a
        next_states = []; step_costs = []; feasible = 0; total = len(belief)
        for (r, c) in belief:
            rr, cc = r + dr, c + dc
            if not in_bounds(rr, cc) or grid[rr][cc]["hole"]: continue
            cst = move_cost(grid[r][c]["h"], grid[rr][cc]["h"])
            if cst is None: continue
            feasible += 1; next_states.append((rr, cc)); step_costs.append(cst)
        if total == 0: return None
        ratio = feasible / total
        if ratio < min_feasible_ratio or not next_states: return None
        next_belief = frozenset(next_states)
        step = (sum(step_costs) / len(step_costs)) if use_avg_cost else max(step_costs)
        return next_belief, step
    def is_goal_belief(belief): return belief and all(s == goal for s in belief)
    pq = PriorityQueue()
    pq.put((0.0, B0)); g_cost = {B0: 0.0}; parent = {B0: None}; parent_action = {B0: None}
    while not pq.empty():
        g, B = pq.get()
        if g > g_cost.get(B, float('inf')) + 1e-12: continue
        if is_goal_belief(B):
            acts = []; curB = B
            while parent[curB] is not None:
                acts.append(parent_action[curB]); curB = parent[curB]
            acts.reverse()
            path = [start]; r, c = start; fuel_used = 0.0
            for (dr, dc) in acts:
                rr, cc = r + dr, c + dc
                if not in_bounds(rr, cc) or grid[rr][cc]["hole"]: return None
                cst = move_cost(grid[r][c]["h"], grid[rr][cc]["h"])
                if cst is None or fuel_used + cst > fuel_budget + 1e-9: return None
                fuel_used += cst; r, c = rr, cc; path.append((r, c))
            return path if path and path[-1] == goal else None
        for a in actions:
            res = apply_action_partial(B, a)
            if res is None: continue
            B2, step = res; ng = g + step
            if ng > fuel_budget + 1e-9: continue
            if B2 not in g_cost or ng < g_cost[B2] - 1e-12:
                g_cost[B2]=ng; parent[B2]=B; parent_action[B2]=a; pq.put((ng, B2))
    return None

# =========================
# BUTTON UI
# =========================
class Button:
    def __init__(self, rect, text, font, on_click):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.on_click = on_click
        self.hover = False
    def draw(self, surface):
        bg = COLOR_BTN_HOVER if self.hover else COLOR_BTN
        pygame.draw.rect(surface, bg, self.rect, border_radius=8)
        pygame.draw.rect(surface, (0,0,0), self.rect, 1, border_radius=8)
        if self.text:
            label = self.font.render(self.text, True, COLOR_BTN_TEXT)
            surface.blit(label, (self.rect.centerx - label.get_width()//2,
                                 self.rect.centery - label.get_height()//2))
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.on_click:
                self.on_click()

# =========================
# MÀN HÌNH NHẬP XĂNG BAN ĐẦU
# =========================
def fuel_input_screen(screen, font, big_font, left_w, play_w, right_w, full_h):
    user_text = ""
    prompt = "Nhập số xăng ban đầu rồi nhấn Enter:"
    warn = ""
    while True:
        screen.fill(COLOR_BG)
        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(0, 0, left_w, full_h))
        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(left_w + play_w, 0, right_w, full_h))
        p_surf = big_font.render("FUEL SETUP", True, COLOR_TEXT)
        screen.blit(p_surf, (left_w + 40, 30))
        t_surf = font.render(prompt, True, COLOR_TEXT)
        screen.blit(t_surf, (left_w + 40, 110))
        box = pygame.Rect(left_w + 40, 150, 360, 40)
        pygame.draw.rect(screen, (60,60,70), box, border_radius=6)
        txt = font.render(user_text, True, COLOR_TEXT)
        screen.blit(txt, (box.x + 10, box.y + 8))
        if warn:
            w_surf = font.render(warn, True, COLOR_WARNING)
            screen.blit(w_surf, (left_w + 40, 200))
        tip = font.render("Esc: thoát", True, (180,180,180))
        screen.blit(tip, (left_w + 40, 240))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
                if event.key == pygame.K_RETURN:
                    try:
                        val = float(user_text)
                        if val <= 0:
                            warn = "Xăng phải > 0."
                        else:
                            return val
                    except:
                        warn = "Vui lòng nhập số hợp lệ (vd: 50 hoặc 75.5)."
                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    ch = event.unicode
                    if ch.isdigit() or ch in ".-":
                        user_text += ch

# =========================
# MAIN GAME LOOP
# =========================
def main():
    pygame.init()
    pygame.display.set_caption("Maze")
    grid, start, goal = find_start_goal(RAW_MAP)

    play_w = COLS * (CELL_SIZE + MARGIN) + MARGIN
    play_h = ROWS * (CELL_SIZE + MARGIN) + MARGIN
    width = LEFTBAR_W + play_w + SIDEBAR_W
    height = max(play_h + 80, 560)
    screen = pygame.display.set_mode((width, height))

    font = pygame.font.SysFont(FONT_NAME, 18)
    big_font = pygame.font.SysFont(FONT_NAME, 28, bold=True)

    fuel = fuel_input_screen(screen, font, big_font, LEFTBAR_W, play_w, SIDEBAR_W, height)
    start_fuel = fuel

    tiles = HeightTiles(CELL_SIZE, MARGIN)
    for h in range(0, 5):
        fname = f"h{h}.png"
        try:
            tiles.load_height_tile(h, fname)
        except Exception:
            pass
    try:
        tiles.load_misc("hole", "water.png")
    except Exception:
        pass

    try:
        home_raw = pygame.image.load("home.png").convert_alpha()
    except Exception:
        home_raw = None

    def blit_home(surface, inner_rect):
        nonlocal home_raw
        if home_raw is None:
            pygame.draw.rect(surface, COLOR_GOAL, inner_rect, border_radius=6)
            return
        scaled = pygame.transform.smoothscale(home_raw, (inner_rect.w, inner_rect.h))
        surface.blit(scaled, inner_rect.topleft)

    car = Car(start, CELL_SIZE)
    visited_cells = {start}

    log_lines = []
    def log(msg):
        log_lines.append(msg)
        if len(log_lines) > 300:
            del log_lines[:150]

    auto_path = None
    auto_idx = 0
    auto_step_ms = 180
    last_step_time = 0
    andor_plan = None
    andor_auto = False
    andor_step_interval_ms = 220
    andor_last_step_ms = 0
    andor_stall_count = 0
    andor_stall_limit = 5

    clock = pygame.time.Clock()

    def cell_rect_local(r, c):
        ox = LEFTBAR_W
        x = ox + MARGIN + c*(CELL_SIZE + MARGIN)
        y = 0 + MARGIN + r*(CELL_SIZE + MARGIN)
        return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

    def set_auto_path(name, path):
        nonlocal auto_path, auto_idx, last_step_time
        if path is None:
            log(f"[{name}] Không tìm được đường.")
        else:
            log(f"[{name}] Tìm được đường dài {len(path)} ô.")
            auto_path = path; auto_idx = 0; last_step_time = 0

    # --- HÀM CHẠY TỪNG THUẬT TOÁN ---
    def run_ac3():
        nonlocal grid, car, visited_cells, auto_path, auto_idx, last_step_time
        removed, keep = ac3_reduce_map(grid, start, goal)
        if (car.r, car.c) not in keep:
            car = Car(start, CELL_SIZE)
        visited_cells = {p for p in visited_cells if p in keep} or {(car.r, car.c)}
        auto_path = None; auto_idx = 0; last_step_time = 0
        log(f"[AC-3] Đã rút gọn bản đồ: loại {removed} ô không có ‘hỗ trợ hai phía’.")

    def run_ids(): set_auto_path("IDS", ids_path(grid, (car.r, car.c), goal, fuel_budget=fuel))
    def run_a_star(): set_auto_path("A*", a_star_path(grid, (car.r, car.c), goal, fuel_budget=fuel))
    def run_hill_climbing(): set_auto_path("Hill", hill_climbing_path(grid, (car.r, car.c), goal))
    def run_bfs(): set_auto_path("BFS", bfs_path(grid, (car.r, car.c), goal, fuel_budget=fuel))
    def run_ucs(): set_auto_path("UCS", ucs_path(grid, (car.r, car.c), goal, fuel_budget=fuel))
    def run_sa():
        path = sa_path(grid, (car.r, car.c), goal)
        if path is not None and not can_follow_path_with_fuel(grid, path, fuel):
            log("không thể tìm được đường đi với lượng nhiên liệu hiện tại")
            path = None
        set_auto_path("SA", path)
    def run_uncertain():
        nonlocal andor_plan, andor_auto, andor_last_step_ms, andor_stall_count
        log("[Uncertain] Đang tạo kế hoạch...")
        # NÊN tạo từ vị trí hiện tại, không phải start:
        andor_plan = and_or_graph_search(grid, (car.r, car.c), goal, depth_limit=600)

        if andor_plan is None:
            log("[Uncertain] Không tạo được kế hoạch.")
            andor_auto = False
        else:
            log("[Uncertain] Đã có kế hoạch. Bắt đầu tự chạy…")
            andor_auto = True
            andor_last_step_ms = 0
            andor_stall_count = 0
    def run_forward_check():
        set_auto_path("ForwardCheck",
                  forward_checking_path(grid, (car.r, car.c), goal, depth_limit=ROWS*COLS, fuel_budget=fuel, on_fail=log))
    def run_belief(): set_auto_path("Belief (Conformant)", belief_conformant_path(grid, (car.r, car.c), goal, fuel_budget=fuel, init_radius=1))
    def run_dfs(): set_auto_path("DFS", dfs_path(grid, (car.r, car.c), goal, fuel_budget=fuel))
    def run_greedy(): set_auto_path("Greedy", greedy_best_first_path(grid, (car.r, car.c), goal, fuel_budget=fuel))

    # >>> Điều chỉnh BEAM: beam_width=20, alpha=0.05, max_iters=200000
    def run_beam():
        set_auto_path("Beam", beam_search_path(
            grid, (car.r, car.c), goal, fuel_budget=fuel,
            beam_width=20,
            alpha=0.05,
            max_iters=200000
        ))

    def run_backtracking(): set_auto_path("Backtracking", backtracking_path(grid, (car.r, car.c), goal, fuel_budget=fuel))

    # >>> Điều chỉnh Belief-Partial: init_radius=0, min_feasible_ratio=0.3, use_avg_cost=True
    def run_belief_partial():
        set_auto_path("Belief-Partial", belief_partial_path(
            grid, (car.r, car.c), goal, fuel_budget=fuel,
            init_radius=0,
            min_feasible_ratio=0.3,
            use_avg_cost=True
        ))

    # --- Reset (định nghĩa TRƯỚC khi tạo nút) ---
    def reset_all():
        nonlocal car, visited_cells, fuel, auto_path, auto_idx, last_step_time
        car = Car(start, CELL_SIZE)
        visited_cells = {start}
        fuel = start_fuel
        auto_path = None; auto_idx = 0
        last_step_time = 0
        log("[Reset] Xe đã trở về điểm xuất phát. Đã xoá dấu đường đi và khôi phục xăng.")

    # ====== NÚT BÊN TRÁI (ĐƯA RESET XUỐNG DƯỚI) ======
    LBTN_X = 16; LBTN_Y = 16; LBTN_W = LEFTBAR_W - 32; LBTN_H = 36; LBTN_GAP = 8
    button_specs = [
        ("AC-3 Reduce", run_ac3),
        ("A*",          run_a_star),
        ("UCS",         run_ucs),
        ("BFS",         run_bfs),
        ("IDS",         run_ids),
        ("Hill Climbing", run_hill_climbing),
        ("Simulated Anneal", run_sa),
        ("Uncertain Action", run_uncertain),
        ("Forward Checking", run_forward_check),
        ("Belief Search", run_belief),
        ("DFS", run_dfs),
        ("Greedy", run_greedy),
        ("Beam", run_beam),
        ("Backtracking", run_backtracking),
        ("Belief-Partial", run_belief_partial),
        ("Reset",       reset_all),        # ⇦ đưa xuống cuối
    ]
    left_buttons = []
    for i, (label, handler) in enumerate(button_specs):
        rect = pygame.Rect(LBTN_X, LBTN_Y + i*(LBTN_H + LBTN_GAP), LBTN_W, LBTN_H)
        left_buttons.append(Button(rect, label, font, handler))

    status_panel_rect_left = pygame.Rect(
        8,
        LBTN_Y + len(left_buttons)*(LBTN_H + LBTN_GAP) + 8,
        LEFTBAR_W - 16,
        STATUS_PANEL_H
    )

    running = True
    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            for b in left_buttons:
                b.handle_event(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # PHÍM TẮT RESET
                if event.key == pygame.K_r:
                    reset_all()
                # Điều khiển xe
                dr = dc = 0
                if event.key in (pygame.K_UP, pygame.K_w): dr = -1
                elif event.key in (pygame.K_DOWN, pygame.K_s): dr = 1
                elif event.key in (pygame.K_LEFT, pygame.K_a): dc = -1
                elif event.key in (pygame.K_RIGHT, pygame.K_d): dc = 1
                if dr or dc:
                    moved, fuel, msg = car.try_move(dr, dc, grid, fuel)
                    if moved:
                        visited_cells.add((car.r, car.c))
                    log(f"[Key] {msg} (fuel {fuel:.1f}/{start_fuel:.1f})")

        # Auto-walk
        if auto_path and auto_idx < len(auto_path):
            last_step_time += dt
            if last_step_time >= auto_step_ms:
                last_step_time = 0
                if (car.r, car.c) == auto_path[auto_idx]:
                    auto_idx += 1
                if auto_idx < len(auto_path):
                    nr, nc = auto_path[auto_idx]
                    dr = nr - car.r; dc = nc - car.c
                    moved, fuel, msg = car.try_move(dr, dc, grid, fuel)
                    if moved:
                        visited_cells.add((car.r, car.c))
                    log(f"{msg} (fuel {fuel:.1f}/{start_fuel:.1f})")
                    log(f"→ vị trí {(car.r,car.c)}")
                    if not moved:
                        log("Dừng: không thể đi tiếp."); auto_path = None
                    else:
                        auto_idx += 1
                        if (car.r, car.c) == goal:
                            log("ĐÃ ĐẾN ĐÍCH!"); auto_path = None
        # --- AND–OR AUTO-RUN ---
        if andor_auto and andor_plan:
            now = pygame.time.get_ticks()
            if andor_last_step_ms == 0 or (now - andor_last_step_ms) >= andor_step_interval_ms:


                cur_state = (car.r, car.c)
                act = next_action_from_plan(andor_plan, cur_state)
           
                if act is None:
                    # Kế hoạch đã SUCCESS tại node hiện tại
                    andor_auto = False
                    log("AND–OR auto: đã hoàn thành kế hoạch.")
                else:
                    # Mô phỏng outcome ~ uncertain_results (không có stay)
                    nxt_state = stochastic_step(grid, cur_state, act)
                    nr, nc = nxt_state


                    # Suy ra hướng thực tế
                    dr, dc = nr - car.r, nc - car.c
                    actual_dir = delta_to_dir(dr, dc)


                    if nxt_state == cur_state:
                        # Không nhúc nhích (hiếm, do outs rỗng -> _try_step đều None)
                        andor_stall_count += 1
                        if andor_stall_count >= andor_stall_limit:
                            andor_auto = False
                            log("AND–OR auto: kẹt liên tiếp, dừng tự chạy.")
                            print("AND–OR auto: kẹt liên tiếp, dừng tự chạy.")
                        else:
                            log(f"AND–OR auto: dự định {act} • thực tế: (stay)")
                            print(f"AND–OR auto: dự định {act} • thực tế: (stay)")
                        andor_last_step_ms = now
                    else:
                        # Dùng try_move để move + trừ xăng + cập nhật hướng một lần
                        dr, dc = nr - car.r, nc - car.c
                        moved, fuel, msg = car.try_move(dr, dc, grid, fuel)
                        if not moved:
                            andor_auto = False
                            log(f"AND–OR auto: {msg}")
                        else:
                            andor_stall_count = 0
                            visited_cells.add((car.r, car.c))
                            log(f"AND–OR auto: dự định {act} • thực tế {actual_dir} ({msg})")
                            print(f"AND–OR auto: dự định {act} • thực tế {actual_dir} ({msg})")


                            # Cập nhật kế hoạch theo outcome thật (vị trí đã là (car.r,car.c) mới)
                            andor_plan = advance_plan_after_outcome(andor_plan, (car.r, car.c))


                            if (car.r, car.c) == goal:
                                andor_auto = False
                                log("AND–OR auto: đã tới đích!")
                                print("AND–OR auto: đã tới đích!")


                        andor_last_step_ms = now



        # ===== VẼ =====
        screen.fill(COLOR_BG)

        # Bảng trái
        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(0, 0, LEFTBAR_W, height))
        pygame.draw.line(screen, COLOR_SEP, (LEFTBAR_W, 0), (LEFTBAR_W, height), 2)
        for b in left_buttons:
            b.draw(screen)
        pygame.draw.rect(screen, COLOR_PANEL, status_panel_rect_left, border_radius=8)
        pygame.draw.rect(screen, COLOR_SEP, status_panel_rect_left, 2, border_radius=8)

        # Khu mê cung (giữa)
        for r in range(ROWS):
            for c in range(COLS):
                rect = cell_rect_local(r, c)
                pygame.draw.rect(screen, COLOR_GRID, rect, border_radius=4)
                inner = rect.inflate(-MARGIN*2, -MARGIN*2)

                cell = grid[r][c]
                if cell["hole"]:
                    hole_tile = tiles.get_misc("hole", inner)
                    if hole_tile is not None:
                        screen.blit(hole_tile, inner.topleft)
                    else:
                        pygame.draw.rect(screen, COLOR_HOLE, inner, border_radius=6)
                    continue

                h_val = cell["h"]
                tile_img = tiles.get_for_height(h_val, inner)
                if tile_img is not None:
                    screen.blit(tile_img, inner.topleft)
                else:
                    pygame.draw.rect(screen, COLOR_FLAT, inner, border_radius=6)

                if cell["goal"]:
                    blit_home(screen, inner)

                if (r, c) == (car.r, car.c):
                    pygame.draw.rect(screen, (255,255,255), inner, width=2, border_radius=8)

                if (r, c) in visited_cells and not cell["goal"] and not cell["start"]:
                    cx = inner.x + inner.w // 2
                    cy = inner.y + inner.h // 2
                    pygame.draw.circle(screen, COLOR_VISITED, (cx, cy), VISITED_DOT_RADIUS)

                h_col = (20, 20, 25) if (r, c) != (car.r, car.c) else (0, 60, 0)
                h_txt = font.render(str(h_val), True, h_col)
                screen.blit(h_txt, (inner.x + 6, inner.y + 4))

        # Vẽ xe cuối
        car_rect = cell_rect_local(car.r, car.c).inflate(-MARGIN*2, -MARGIN*2)
        car.draw(screen, car_rect)

        # Bảng phải (log)
        right_x = LEFTBAR_W + play_w
        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(right_x, 0, SIDEBAR_W, height))
        pygame.draw.line(screen, COLOR_SEP, (right_x, 0), (right_x, height), 2)
        title = big_font.render("TRẠNG THÁI BƯỚC ĐI", True, COLOR_TEXT)
        screen.blit(title, (right_x + (SIDEBAR_W - title.get_width())//2, 8))
        log_rect = pygame.Rect(right_x + 12, 48, SIDEBAR_W - 24, height - 60)
        pygame.draw.rect(screen, (24,24,28), log_rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_SEP, log_rect, 2, border_radius=8)
        y = log_rect.bottom - 8
        for line in reversed(log_lines):
            surf = font.render(line, True, COLOR_TEXT)
            y -= surf.get_height() + 4
            if y < log_rect.y + 8:
                break
            screen.blit(surf, (log_rect.x + 8, y))

        draw_fuel_gauge(screen, status_panel_rect_left, fuel, start_fuel, font, title="NHIÊN LIỆU")

        pygame.display.flip()

    pygame.quit()

# =========================
# HOOKS CHO AI SAU NÀY
# =========================
def is_goal(state, goal):
    return state == goal

def get_neighbors(state):
    r, c = state
    return [(rr,cc) for rr,cc in neighbors4(r,c)]

def step_cost(grid, a, b):
    ar, ac = a; br, bc = b
    if grid[br][bc]["hole"]:
        return None
    return move_cost(grid[ar][ac]["h"], grid[br][bc]["h"])

def admissible_heuristic(h_current, h_goal):
    return 0.0

if __name__ == "__main__":
    main()
