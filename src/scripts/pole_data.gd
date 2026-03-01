class_name PoleData
extends Resource

@export var poleID : String
@export var pics : Array[Image]
@export var poleType : String
@export var surroundings : Array[String]
@export var vegetation : int

# AI result fields — populated by the backend after photo submission
@export var ai_pole_type : String
@export var ai_components : Array[String]
@export var ai_encroachment : bool
@export var ai_annotated_image : Image

static func create_empty_pole() -> PoleData:
	var p = PoleData.new()
	p.poleID = ""
	var emptyimg : Array[Image] = []
	p.pics = emptyimg
	p.poleType = ""
	var emptystrings : Array[String] = []
	p.surroundings = emptystrings
	p.vegetation = 0
	p.ai_pole_type = ""
	var empty_ai_components : Array[String] = []
	p.ai_components = empty_ai_components
	p.ai_encroachment = false
	p.ai_annotated_image = null
	return p

static func create_from_pics(pics:Array[Image]) -> PoleData:
	var p = create_empty_pole()
	p.pics = pics
	return p
