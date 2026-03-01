extends Control

@export var pole_data : PoleData
@export var texture_rects : Array[TextureRect]
@export var id_text : TextEdit
@export var type_text : TextEdit
@export var transformer_box : CheckBox
@export var insulator_box : CheckBox
@export var streetlight_box : CheckBox
@export var vegetation_options : OptionButton

const main_menu = preload("uid://34b6yye70imh")

func _ready() -> void:
	if GlobalData.pole_data:
		pole_data = GlobalData.pole_data
		update_ui()

func update_ui():
	if pole_data:
		var i = 0
		while i < min(pole_data.pics.size(), 2):
			texture_rects[i].texture = ImageTexture.create_from_image(pole_data.pics[i])
			i += 1
		id_text.text = pole_data.poleID
		type_text.text = pole_data.poleType
		transformer_box.button_pressed = pole_data.surroundings.has("transformer")
		insulator_box.button_pressed = pole_data.surroundings.has("insulator")
		streetlight_box.button_pressed = pole_data.surroundings.has("streetlight")
		vegetation_options.select(pole_data.vegetation)

func update_data():
	pole_data.poleID = id_text.text
	pole_data.poleType = pole_data.poleType
	pole_data.surroundings = []
	if transformer_box.button_pressed:
		pole_data.surroundings.append("transformer")
	if insulator_box.button_pressed:
		pole_data.surroundings.append("insulator")
	if streetlight_box.button_pressed:
		pole_data.surroundings.append("streetlight")
	pole_data.vegetation = vegetation_options.selected


func _on_button_pressed() -> void:
	update_data()
	#get_tree().root.add_child(main_menu.instantiate())
	queue_free()
