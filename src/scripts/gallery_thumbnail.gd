extends Control

@export var pole_data : PoleData

@export var texture_rect : TextureRect
@export var text_label : RichTextLabel

const DATA_DIR = "user://data/"
const FORM_SCENE = preload("uid://y33a508r54up")


func _process(delta: float) -> void:
	if texture_rect.texture == null and text_label.text != "":
		pole_data = ResourceLoader.load(DATA_DIR + "pole_" + text_label.text + ".res")
		update_ui()

func update_ui():
	if pole_data.pics.size() > 0:
		texture_rect.texture = ImageTexture.create_from_image(pole_data.pics[0])
	text_label.text = "[center]" + pole_data.poleID


func _on_texture_rect_gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		GlobalData.pole_data = pole_data
		get_tree().root.add_child(FORM_SCENE.instantiate())
		queue_free()


func set_pole_data(pole_data:PoleData):
	self.pole_data = pole_data
