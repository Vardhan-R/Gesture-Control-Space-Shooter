import os
# import warnings

# warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2, HandTrackingModule as htm, import_vectors as vect, numpy as np

class Enemy:
	def __init__(self, max_health: int) -> None:
		self.max_health = max_health
		self.health = self.max_health
		self.radius = 15
		self.pos = vect.Vector(np.random.randint(self.radius + 1, width - self.radius - 2), -self.radius)
		self.vel = vect.Vector(0, 2)
		self.status = "alive"

	def show(self, img) -> None:
		cv2.circle(img, (self.pos.x, self.pos.y), self.radius, (0, 255, 0), -1)
		cv2.rectangle(img, (self.pos.x - self.radius, self.pos.y - self.radius - 2), (self.pos.x + self.radius, self.pos.y - self.radius - 7), (0, 255, 0), 1)
		cv2.rectangle(img, (self.pos.x - self.radius, self.pos.y - self.radius - 2), (round(self.pos.x + (2 * self.health / self.max_health - 1) * self.radius), self.pos.y - self.radius - 7), (0, 255, 0), -1)

	def update(self, mode: str, proximal_vect: vect.Vector, distal_vect: vect.Vector, target_y: int, damage: int = 1) -> None:
		match mode:
			case "base":
				a, b, c = proximal_vect.y - distal_vect.y, distal_vect.x - proximal_vect.x, proximal_vect.x * distal_vect.y - distal_vect.x * proximal_vect.y
				beam_vect = distal_vect - proximal_vect
				if abs(a * self.pos.x + b * self.pos.y + c) / vect.Vector(a, b).mag() <= self.radius and vect.angBetween(beam_vect, self.pos - distal_vect) < np.pi / 2:
					self.health -= damage
			case "triple":
				proximal_vect = (proximal_vect - distal_vect).rotate(-np.pi / 9) + distal_vect
				for _ in range(3):
					proximal_vect = (proximal_vect - distal_vect).rotate(np.pi / 18) + distal_vect
					a, b, c = proximal_vect.y - distal_vect.y, distal_vect.x - proximal_vect.x, proximal_vect.x * distal_vect.y - distal_vect.x * proximal_vect.y
					beam_vect = distal_vect - proximal_vect
					if abs(a * self.pos.x + b * self.pos.y + c) / vect.Vector(a, b).mag() <= self.radius and vect.angBetween(beam_vect, self.pos - distal_vect) < np.pi / 2:
						self.health -= damage
		if self.health <= 0:
			# return "dead"
			self.status = "dead"
		self.pos += self.vel
		if self.pos.y + self.radius >= target_y:
			# return "crossed"
			self.status = "crossed"

def dispBasicScreen(title: str, detector_indices: list[int] = [8], back_button_scale: float | int = 0) -> None:
	global click_val, game_status, img

	finger_pos = detector.getPosition(img, detector_indices, draw=False)
	if finger_pos:
		finger_vect = vect.Vector(*finger_pos[0])
	img = np.zeros_like(img)

	cv2.putText(img, title, (base_width, base_height), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255))

	if back_button_scale > 0:
		cv2.putText(img, "<", (5, 55), cv2.FONT_HERSHEY_COMPLEX, back_button_scale, (0, 255, 255))
		cv2.rectangle(img, (5, 5), (65, 65), (0, 255, 0), 1)

		if finger_pos:
			if 5 < finger_vect.x < 65 and 5 < finger_vect.y < 65:
				click_val += 0.05
			else:
				click_val = 0

			if click_val >= 1:
				game_status = "menu"
				click_val = 0

		cv2.circle(img, finger_pos[0], 10, (255, 0, 0), -1)
		dispClicker(finger_vect)

def dispBeams(mode: str, proximal_vect: vect.Vector, distal_vect: vect.Vector) -> None:
	match mode:
		case "base":
			beam_vect = distal_vect - proximal_vect
			beam_ang = beam_vect.dir()

			if -3 * np.pi / 4 <= beam_ang < -np.pi / 4: # top
				end_pt = (round((distal_vect.x - proximal_vect.x) / (distal_vect.y - proximal_vect.y) * (0 - proximal_vect.y) + proximal_vect.x), 0)
			elif -np.pi / 4 <= beam_ang < np.pi / 4: # right
				end_pt = (width, round((distal_vect.y - proximal_vect.y) / (distal_vect.x - proximal_vect.x) * (width - proximal_vect.x) + proximal_vect.y))
			elif np.pi / 4 <= beam_ang < 3 * np.pi / 4: # down
				end_pt = (round((distal_vect.x - proximal_vect.x) / (distal_vect.y - proximal_vect.y) * (height - proximal_vect.y) + proximal_vect.x), height)
			else: # left
				end_pt = (0, round((distal_vect.y - proximal_vect.y) / (distal_vect.x - proximal_vect.x) * (0 - proximal_vect.x) + proximal_vect.y))

			cv2.line(img, (distal_vect.x, distal_vect.y), end_pt, (128, 128, 128), 2)
		case "triple":
			proximal_vect = (proximal_vect - distal_vect).rotate(-np.pi / 9) + distal_vect
			for _ in range(3):
				proximal_vect = (proximal_vect - distal_vect).rotate(np.pi / 18) + distal_vect

				beam_vect = distal_vect - proximal_vect
				beam_ang = beam_vect.dir()

				if -3 * np.pi / 4 <= beam_ang < -np.pi / 4: # top
					end_pt = (round((distal_vect.x - proximal_vect.x) / (distal_vect.y - proximal_vect.y) * (0 - proximal_vect.y) + proximal_vect.x), 0)
				elif -np.pi / 4 <= beam_ang < np.pi / 4: # right
					end_pt = (width, round((distal_vect.y - proximal_vect.y) / (distal_vect.x - proximal_vect.x) * (width - proximal_vect.x) + proximal_vect.y))
				elif np.pi / 4 <= beam_ang < 3 * np.pi / 4: # down
					end_pt = (round((distal_vect.x - proximal_vect.x) / (distal_vect.y - proximal_vect.y) * (height - proximal_vect.y) + proximal_vect.x), height)
				else: # left
					end_pt = (0, round((distal_vect.y - proximal_vect.y) / (distal_vect.x - proximal_vect.x) * (0 - proximal_vect.x) + proximal_vect.y))

				cv2.line(img, (distal_vect.x, distal_vect.y), end_pt, (128, 128, 128), 2)

def dispClicker(pos: vect.Vector) -> None:
	if click_val:
		cv2.ellipse(img, (pos.x, pos.y), (17, 17), 0, -90, -90 + 360 * click_val, (0, 100, 0), 2)

def doGameOverStuff() -> None:
	global click_val, enemies, game_over_time, game_status, img, mode, n_enemies, nuke_val, spawn_rate

	img = np.zeros_like(img)

	for enemy in enemies:
		if enemy.status == "crossed":
			enemy.show(img)

	cv2.line(img, (0, target_y), (width, target_y), (0, 0, 255), 1)
	cv2.putText(img, "GAME OVER", (base_width, base_height), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255))
	cv2.putText(img, str(score), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))

	game_over_time += 0.01

	if game_over_time >= 1:
		enemies = []
		n_enemies = len(enemies)
		nuke_val = 1
		click_val = 0
		game_over_time = 0
		spawn_rate = 0.5
		mode = "base"

		if len(leaderboard) < 5:
			game_status = "save_score"
			return

		least_high_score = int(leaderboard[-1].split(' ')[-1][:-1]) # [ABC 420\n, DEF 69\n, GHI 37\n]
		if score > least_high_score:
			game_status = "save_score"
			return

		game_status = "menu"

def doLeaderboardStuff() -> None:
	global click_val, game_status, img

	finger_pos = detector.getPosition(img, [8], draw=False)
	if finger_pos:
		finger_vect = vect.Vector(*finger_pos[0])
	img = np.zeros_like(img)

	cv2.putText(img, "<", (5, 55), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 255, 255))
	cv2.rectangle(img, (5, 5), (65, 65), (0, 255, 0), 1)
	cv2.putText(img, "LEADERBOARD", (base_width, base_height), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255))
	for i, entry in enumerate(leaderboard):
		cv2.putText(img, f"{i + 1}. {entry[:-1]}", (base_width, base_height + (i + 1) * diff), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))

	if finger_pos:
		if 5 < finger_vect.x < 65 and 5 < finger_vect.y < 65:
			click_val += 0.05
		else:
			click_val = 0

		if click_val >= 1:
			game_status = "menu"
			click_val = 0

		cv2.circle(img, finger_pos[0], 10, (255, 0, 0), -1)
		dispClicker(finger_vect)

def doMenuStuff() -> None:
	global click_val, curr_selection, game_status, img, prev_selection, score

	score = 0

	finger_pos = detector.getPosition(img, [8], draw=False)
	if finger_pos:
		finger_vect = vect.Vector(*finger_pos[0])
	img = np.zeros_like(img)

	cv2.putText(img, "Space Shooter", (base_width, base_height), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255))

	curr_selection = None
	for i in range(1, 5):
		cv2.putText(img, selections[i - 1], (base_width, base_height + i * diff), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
		cv2.rectangle(img, (base_width, base_height + 15 + (i - 1) * diff), (base_width + 250, base_height + 15 + i * diff), (0, 255, 0), 1)
		if finger_pos:
			if base_width < finger_vect.x < base_width + 250 and base_height + 15 + (i - 1) * diff < finger_vect.y < base_height + 15 + i * diff:
				curr_selection = selections[i - 1]

	if curr_selection == prev_selection != None:
		click_val += 0.05
	else:
		click_val = 0
		prev_selection = curr_selection

	if click_val >= 1:
		game_status = selection_status_pairs[curr_selection]
		click_val = 0

	if finger_pos:
		cv2.circle(img, finger_pos[0], 10, (255, 0, 0), -1)
		dispClicker(finger_vect)

def doPausedStuff() -> None:
	global game_status, img

	finger_pos = detector.getPosition(img, [6, 8, 20], draw=False)
	if len(finger_pos) == 3:
		game_status = "playing"
	img = np.zeros_like(img)
	cv2.line(img, (0, target_y), (width, target_y), (0, 0, 255), 1)
	cv2.putText(img, "PAUSED", (width // 2 - 100, height // 2 - 15), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255))
	cv2.putText(img, str(score), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))

def doPlayingStuff() -> None:
	global game_status, img, mode, spawn_rate

	if not success:
		return

	finger_pos = detector.getPosition(img, [6, 8, 20], draw=False)
	img = np.zeros_like(img)
	if len(finger_pos) != 3:
		game_status = "paused"
		return

	proximal, distal, little = finger_pos
	proximal_vect, distal_vect, little_vect = vect.Vector(*proximal), vect.Vector(*distal), vect.Vector(*little)

	cv2.circle(img, proximal, 10, (255, 0, 0), -1)
	cv2.circle(img, distal, 10, (255, 0, 0), -1)

	dispBeams(mode, proximal_vect, distal_vect)

	updateEnemies(mode, proximal_vect, distal_vect, target_y)

	spawner(spawn_rate)

	if 5 <= score < 10:
		mode = "triple"
	elif 10 <= score:
		spawn_rate = 1
		updateNuke(nuke_vect, little_vect)
		cv2.circle(img, little, 10, (255, 0, 0), -1)
		dispClicker(little_vect)

	cv2.line(img, (0, target_y), (width, target_y), (0, 0, 255), 1)
	cv2.putText(img, str(score), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))

def doQuittingStuff() -> None:
	global img, running

	img = np.zeros_like(img)
	running = False

def doSaveScoreStuff() -> None:
	global game_status, img, leaderboard, score

	img = np.zeros_like(img)

	high_scores = [int(entry.split(' ')[-1][:-1]) for entry in leaderboard] # [ABC 420\n, DEF 69\n, GHI 37\n]

	inserted = False
	for i, high_score in enumerate(high_scores):
		if score > high_score:
			leaderboard.insert(i, f"{''.join([chr(x) for x in np.random.randint(65, 90, 3)])} {score}\n")
			if len(leaderboard) > 5:
				leaderboard.pop()
			inserted = True
			break

	if not inserted and len(leaderboard) < 5:
		leaderboard.append(f"{''.join([chr(x) for x in np.random.randint(65, 90, 3)])} {score}\n")

	with open("./leaderboard.txt", 'w') as file_ptr:
		file_ptr.writelines(leaderboard)

	score = 0

	game_status = "leaderboard"

def doSettingsStuff() -> None:
	global click_val, game_status, img

	finger_pos = detector.getPosition(img, [8], draw=False)
	if finger_pos:
		finger_vect = vect.Vector(*finger_pos[0])
	img = np.zeros_like(img)

	cv2.putText(img, "<", (5, 55), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 255, 255))
	cv2.rectangle(img, (5, 5), (65, 65), (0, 255, 0), 1)
	cv2.putText(img, "SETTINGS", (base_width, base_height), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255))

	if finger_pos:
		if 5 < finger_vect.x < 65 and 5 < finger_vect.y < 65:
			click_val += 0.05
		else:
			click_val = 0

		if click_val >= 1:
			game_status = "menu"
			click_val = 0

	cv2.circle(img, finger_pos[0], 10, (255, 0, 0), -1)
	dispClicker(finger_vect)

def spawner(spawn_rate: int = 1) -> None:
	global n_enemies

	if np.random.random() < np.exp(-n_enemies / spawn_rate):
		enemies.append(Enemy(20))
		n_enemies += 1

def updateEnemies(mode: str, proximal_vect: vect.Vector, distal_vect: vect.Vector, target_y: int) -> None:
	global game_status, n_enemies, score

	for i, enemy in enumerate(enemies.copy()):
		enemy.update(mode, proximal_vect, distal_vect, target_y)
		match enemy.status:
			case "dead":
				score += 1
				del enemies[i]
				n_enemies -= 1
				continue
			case "crossed":
				game_status = "game_over"
		enemy.show(img)

def updateNuke(nuke_vect: vect.Vector, little_vect: vect.Vector) -> None:
	global click_val, enemies, n_enemies, nuke_val, score

	if (little_vect - nuke_vect).magSq() <= 225 and nuke_val >= 1:
		click_val += 0.1
	else:
		click_val = 0
	if click_val >= 1:
		score += n_enemies
		enemies = []
		n_enemies = 0
		nuke_val = 0
		click_val = 0
	nuke_val += 0.001

	cv2.ellipse(img, (nuke_vect.x, nuke_vect.y), (15, 15), 0, -90, -90 + 360 * nuke_val, (0, 128, 255), -1)

all_game_statuses = {"menu": "doMenuStuff()",
					 "leaderboard": "doLeaderboardStuff()",
					 "settings": "doSettingsStuff()",
					 "quitting": "doQuittingStuff()",
					 "playing": "doPlayingStuff()",
					 "paused": "doPausedStuff()",
					 "game_over": "doGameOverStuff()",
					 "save_score": "doSaveScoreStuff()"}
all_modes = ["base", "triple"]
all_power_ups = ["nuke"]

running = True
cv2.namedWindow("Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(0)
success, img = cap.read()
height, width = img.shape[:2]

detector = htm.FindHands()

game_status = "menu"
selections = ["START GAME", "LEADERBOARD", "SETTINGS", "QUIT"]
selection_status_pairs = {"START GAME": "playing", "LEADERBOARD": "leaderboard", "SETTINGS": "settings", "QUIT": "quitting"}
prev_selection, curr_selection = None, None
base_width = width // 4
base_height = 120
diff = 40
enemies = []
n_enemies = len(enemies)
spawn_rate = 0.5
target_y = height - 50
score = 0
mode = "base"
nuke_vect = vect.Vector(width - 20, height // 2)
nuke_val = 1
click_val = 0
game_over_time = 0

with open("./leaderboard.txt", 'r') as file_ptr:
	leaderboard = file_ptr.readlines()

while running:
	success, img = cap.read()
	img = cv2.flip(img, 1)

	eval(all_game_statuses[game_status])

	cv2.imshow("Game", img)

	if chr(cv2.waitKey(1) & 0xFF) == 'q':
		break

'''
make the screen size better
v leaderboard
settings
v game over
v save scores
add a keyboard for entering a name for a high score
enemy types (fast with low health)
coloured enemies
more modes
more power-ups
add music and sound effects
use images instead of circles
add instructions

bugs:
v other enemies may flicker when an enemy dies
v beam backshots
v long beam division by zero
v score is not being reset to zero if the leaderboard is not updated
'''
