extends Control

@export var thumbnail_root : Control
@export var thumnail_tscn : PackedScene
var poles : Array[PoleData]
const DATA_DIR = "user://data/"
const COL_COUNT = 3
const COL_SPACING = 260
const ROW_SPACING = 300

func _ready() -> void:
	update_ui()

func update_ui() -> void:
	for child in thumbnail_root.get_children():
		child.queue_free()
	var ls = ResourceLoader.list_directory(DATA_DIR)
	for path in ls:
		poles.append( ResourceLoader.load(DATA_DIR + path) )
	var i = 0
	print(poles)
	for pole in poles:
		var t = thumnail_tscn.instantiate()
		print(t)
		t.position = Vector2(COL_SPACING*(i%COL_COUNT), ROW_SPACING*(i/COL_COUNT))
		t.get_child(1).text = pole.poleID
		thumbnail_root.add_child(t)
		i += 1
