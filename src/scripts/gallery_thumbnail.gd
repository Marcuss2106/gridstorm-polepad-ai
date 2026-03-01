extends Control

@export var pole_data : PoleData

@export var texture_rect : TextureRect
@export var text_label : RichTextLabel

func _init(pole_data) -> void:
	self.pole_data = pole_data

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	if pole_data:
		update_ui()

func update_ui():
	if pole_data.pics.size() > 0:
		texture_rect.texture = ImageTexture.create_from_image(pole_data.pics[0])
	text_label.text = "[center]" + pole_data.poleID


func _on_texture_rect_gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		print("clickkkkk")
