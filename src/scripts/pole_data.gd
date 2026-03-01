class_name PoleData
extends Resource

@export var poleID : String
@export var pics : Array[Image]
@export var poleType : String
@export var surroundings : Array[String]
@export var vegetation : int

static func create_empty_pole() -> PoleData:
	var p = PoleData.new()
	p.poleID = ""
	var emptyimg : Array[Image] = []
	p.pics = emptyimg
	p.poleType = ""
	var emptystrings : Array[String] = []
	p.surroundings = emptystrings
	p.vegetation = -1
	return p

static func create_from_pics(pics:Array[Image]) -> PoleData:
	var p = create_empty_pole()
	p.pics = pics
	return p
