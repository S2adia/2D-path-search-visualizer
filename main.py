import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
from grid import generate_grid, build_graph, place_start_goal, is_reachable
from search import get_algorithm, reconstruct_path
from visualizer import draw_grid, draw_tree, draw_result_banner

#app Constants
APP_TITLE = "2D Path Search Visualizer"
BG_COLOR = "#F5F5F5"
PANEL_COLOR = "#FFFFFF"
ACCENT = "#5C6BC0"
TEXT_COLOR = "#212121"
BTN_RUN = "#388E3C"
BTN_RESET = "#C62828"
BTN_STEP = "#1565C0"


class PathSearchApp:
    #Main application class managing GUI, state, and animation
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(True, True)
        #hndle window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        #internal state
        #active generator
        self.search_gen = None  
        #auto-play active
        self.running = False  
        #step mode waiting
        self.paused = False  
        #search finished
        self.done = False  
        self.step_event = threading.Event()
        self.thread = None
        self.grid = None
        self.graph = None
        self.start = None
        self.goal = None
        self.visited = set()
        self.frontier = []
        self.parent_map = {}
        self.current_node = None
        self.path = []
        self.step_count = 0

        self._build_ui()
        self._new_grid()

    #UI construction
    def _build_ui(self):
        #build the tkinter layout
        #left control panel
        ctrl = tk.Frame(self.root, bg=PANEL_COLOR, width=220, padx=14, pady=14)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)
        #title
        tk.Label(
            ctrl,
            text="⬡ PATH SEARCH",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=("Courier", 13, "bold"),
        ).pack(pady=(4, 16))
        self._section(ctrl, "GRID SETTINGS")
        #grid size slider
        self._label(ctrl, "Grid Size (N×N)")
        self.var_size = tk.IntVar(value=10)
        self._slider(ctrl, self.var_size, 5, 25, 1)
        self.lbl_size = self._value_label(ctrl, self.var_size, suffix="×")
        self.var_size.trace_add(
            "write", lambda *_: self._update_label(self.lbl_size, self.var_size, "×")
        )
        #obstacle % slider
        self._label(ctrl, "Obstacle Density")
        self.var_obs = tk.DoubleVar(value=25.0)
        self._slider(ctrl, self.var_obs, 20, 30, 1)
        self.lbl_obs = self._value_label(ctrl, self.var_obs, suffix="%", fmt=".0f")
        self.var_obs.trace_add(
            "write",
            lambda *_: self._update_label(self.lbl_obs, self.var_obs, "%", ".0f"),
        )

        self._section(ctrl, "ANIMATION")
        #speed slider
        self._label(ctrl, "Speed (delay ms)")
        self.var_speed = tk.IntVar(value=200)
        self._slider(ctrl, self.var_speed, 10, 1000, 10)
        self.lbl_speed = self._value_label(ctrl, self.var_speed, suffix="ms")
        self.var_speed.trace_add(
            "write", lambda *_: self._update_label(self.lbl_speed, self.var_speed, "ms")
        )

        #step mode toggle
        self.var_step = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ctrl,
            text="Step Mode",
            variable=self.var_step,
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            selectcolor=ACCENT,
            activebackground=PANEL_COLOR,
            activeforeground=TEXT_COLOR,
            font=("Courier", 10),
        ).pack(pady=6)

        self._section(ctrl, "CONTROLS")

        #buttons
        self.btn_run = tk.Button(
            ctrl,
            text="START",
            bg=BTN_RUN,
            fg="white",
            font=("Courier", 11, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
            cursor="hand2",
            command=self._on_start,
        )
        self.btn_run.pack(fill=tk.X, pady=4)
        self.btn_step = tk.Button(
            ctrl,
            text="NEXT STEP",
            bg=BTN_STEP,
            fg="white",
            font=("Courier", 10, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
            command=self._on_step,
        )
        self.btn_step.pack(fill=tk.X, pady=4)
        self.btn_reset = tk.Button(
            ctrl,
            text="NEW GRID",
            bg=BTN_RESET,
            fg="white",
            font=("Courier", 10, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
            cursor="hand2",
            command=self._new_grid,
        )
        self.btn_reset.pack(fill=tk.X, pady=4)

        #status label
        self._section(ctrl, "STATUS")
        self.lbl_status = tk.Label(
            ctrl,
            text="Ready",
            bg=PANEL_COLOR,
            fg="#2E7D32",
            font=("Courier", 9),
            wraplength=190,
            justify=tk.LEFT,
        )
        self.lbl_status.pack(pady=4)

        #info labels
        self.lbl_info = tk.Label(
            ctrl,
            text="",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=("Courier", 8),
            justify=tk.LEFT,
        )
        self.lbl_info.pack(pady=4)
        #right matplotlib canvas
        canvas_frame = tk.Frame(self.root, bg=BG_COLOR)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.fig, (self.ax_grid, self.ax_tree) = plt.subplots(
            1, 2, figsize=(12, 6), facecolor="#F5F5F5"
        )
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    #widget helpers
    def _section(self, parent, text):
        tk.Label(
            parent, text=text, bg=PANEL_COLOR, fg=ACCENT, font=("Courier", 8, "bold")
        ).pack(anchor=tk.W, pady=(10, 2))
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=2)

    def _label(self, parent, text):
        tk.Label(
            parent, text=text, bg=PANEL_COLOR, fg=TEXT_COLOR, font=("Courier", 9)
        ).pack(anchor=tk.W)

    def _slider(self, parent, var, from_, to, resolution):
        tk.Scale(
            parent,
            variable=var,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            troughcolor="#BDBDBD",
            highlightbackground=PANEL_COLOR,
            sliderrelief=tk.FLAT,
            showvalue=False,
            length=180,
        ).pack(fill=tk.X)

    def _value_label(self, parent, var, suffix="", fmt="d"):
        lbl = tk.Label(parent, bg=PANEL_COLOR, fg="#616161", font=("Courier", 8))
        lbl.pack(anchor=tk.E)
        self._update_label(lbl, var, suffix, fmt)
        return lbl

    @staticmethod
    def _update_label(lbl, var, suffix="", fmt="d"):
        val = var.get()
        lbl.config(text=f"{val:{fmt}}{suffix}")

    # grid and search state
    def _new_grid(self):
        #generate a fresh grid and reset search state
        self._stop_search()
        n = self.var_size.get()
        obs_pct = self.var_obs.get() / 100.0
        self.grid = generate_grid(n, obs_pct)
        self.graph = build_graph(self.grid)
        self.start, self.goal = place_start_goal(self.grid)
        reachable = is_reachable(self.graph, self.start, self.goal)
        self.visited = set()
        self.frontier = []
        self.parent_map = {}
        self.current_node = self.start
        self.path = []
        self.step_count = 0
        self.done = False
        status = "Ready  (solvable)" if reachable else "Ready  (may be unsolvable)"
        self._set_status(status, "#2E7D32" if reachable else "#B71C1C")
        self._set_info("")

        self.btn_run.config(state=tk.NORMAL, text="START")
        self.btn_step.config(state=tk.DISABLED)
        self._draw()

    def _stop_search(self):
        #stop the running animation thread
        self.running = False
        #unblock a waiting step
        self.step_event.set()  
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.search_gen = None
        self.paused = False
        self.done = False

    #button Handlers
    def _on_start(self):
        if self.done:
            self._new_grid()
            return
        
        if self.running:
            #pause auto-play
            self.running = False
            self.btn_run.config(text="RESUME")
            if self.var_step.get():
                self.btn_step.config(state=tk.NORMAL)
            return

        #start or resume
        if self.search_gen is None:
            algo_fn = get_algorithm("BFS")
            self.search_gen = algo_fn(self.graph, self.start, self.goal)
            self.visited = set()
            self.parent_map = {self.start: None}
            self.step_count = 0

        self.running = True
        self.paused = False
        self.btn_run.config(text="⏸  PAUSE")
        if self.var_step.get():
            self.btn_step.config(state=tk.NORMAL)
        else:
            self.btn_step.config(state=tk.DISABLED)

        self._set_status("Running BFS…", "#FFF176")
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _on_step(self):
        #advance one step in step mode
        self.step_event.set()

    #animation loop
    def _run_loop(self):
        #background thread executing search algorithm step by step
        while self.running:
            try:
                current, visited, frontier, parent = next(self.search_gen)
            except StopIteration:
                self.running = False
                self.root.after(0, self._on_search_done)
                return

            self.current_node = current
            self.visited = visited
            self.frontier = frontier
            self.parent_map = parent
            self.step_count += 1
            self.root.after(0, self._draw)
            self.root.after(
                0,
                self._set_info,
                f"Step:      {self.step_count}\n"
                f"Explored:  {len(self.visited)}\n"
                f"Frontier:  {len(self.frontier)}\n"
                f"Current:   {current}",
            )

            if current == self.goal:
                self.running = False
                self.root.after(0, self._on_search_done)
                return

            if self.var_step.get():
                #wait for user to click next step
                self.step_event.clear()
                self.step_event.wait()
            else:
                delay = self.var_speed.get() / 1000.0
                time.sleep(delay)

    def _on_search_done(self):
        #handle search completion
        self.done = True
        self.btn_run.config(text="↺  DONE", state=tk.NORMAL)
        self.btn_step.config(state=tk.DISABLED)
        found = self.goal in self.parent_map
        self.path = reconstruct_path(self.parent_map, self.goal) if found else []
        if found:
            self._set_status(
                f"✓ Goal found!\n"
                f"Path length: {len(self.path) - 1} steps\n"
                f"Explored: {len(self.visited)} states",
                "#2E7D32",
            )
        else:
            self._set_status(
                f"✗ No path found.\nExplored: {len(self.visited)} states.", "#B71C1C"
            )

        self._draw(final=True)

    #drawing
    def _draw(self, final=False):
        #redraw both subplots
        draw_grid(
            self.ax_grid,
            self.grid,
            self.start,
            self.goal,
            self.current_node or self.start,
            self.visited,
            frontier=self.frontier,
            path=self.path if final else [],
            step_count=self.step_count,
            states_explored=len(self.visited),
        )
        draw_tree(
            self.ax_tree,
            self.parent_map,
            self.current_node or self.start,
            self.start,
            path=self.path if final else [],
        )

        if final:
            draw_result_banner(
                self.ax_grid,
                found=bool(self.path),
                path_length=len(self.path) - 1 if self.path else 0,
                states_explored=len(self.visited),
            )

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    #status helpers
    def _set_status(self, text, color="#2E7D32"):
        self.lbl_status.config(text=text, fg=color)

    def _set_info(self, text):
        self.lbl_info.config(text=text)

    def _on_closing(self):
        """Handle window close event - cleanup and exit."""
        self._stop_search()
        self.root.quit()
        self.root.destroy()

def main():
    root = tk.Tk()
    root.geometry("1100x680")
    app = PathSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()