#
#
# // Copied from Python code pascal_voc_io.py
# class xmlVocAnnotations(object):
# 	private _foldername
# 	private _filename
# 	private _databaseSrc
# 	private _imgSize
# 	private _localImgPath
# 	private _boxlist
#
#     function __construct( foldername, filename, imgSize, databaseSrc="Unknown", localImgPath=null)
# 	{
#         self._foldername = foldername
#         self._filename = filename
#         self._databaseSrc = databaseSrc
#         self._imgSize = imgSize
#         self._boxlist = []
#         self._localImgPath = localImgPath
#
# 		self._domDoc = DOMDocument
# 	}
#
#     def prettify(self):
# 	{
#         self._domDoc->formatOutput = true
# 	}
#
#     def genXML(self):
# 	{
#         //
#         // Return XML root
#         //
#         // Check conditions
#
#         if ( (self._filename == null) ||
#                 (self._foldername == null) ||
#                 (self._imgSize == null) ||
#                 (count(self._boxlist) <= 0) )
# 		{
#
# 			return null
# 		}
#
# 		top = self._domDoc->createElement('annotation')
# 		topNode = self._domDoc->appendChild(top)
#
#         folder = self._domDoc->createElement('folder',self._foldername)
# 		folderNode = topNode->appendChild(folder)
#
# 		filename = self._domDoc->createElement('filename',self._filename)
# 		filenameNode = topNode->appendChild(filename)
#
# 		localImgPath = self._domDoc->createElement('path',self._localImgPath)
# 		localImgPathNode = topNode->appendChild(localImgPath)
#
# 		source = self._domDoc->createElement('source')
# 		sourceNode = topNode->appendChild(source)
#
# 		database = self._domDoc->createElement('database',self._databaseSrc)
# 		sourceNode->appendChild(database)
#
# 		size_part = self._domDoc->createElement('size_part')
# 		size_partNode = topNode->appendChild(size_part)
#
#         width  = self._domDoc->createElement('width',  strval(self._imgSize['width']))
# 		height = self._domDoc->createElement('height', strval(self._imgSize['height']))
#         depth  = self._domDoc->createElement('depth',  strval(self._imgSize['depth']))
#
# 		size_partNode->appendChild(width)
# 		size_partNode->appendChild(height)
# 		size_partNode->appendChild(depth)
#
# 		segmented = self._domDoc->createElement("segmented","0")
# 		topNode->appendChild(segmented)
#
#         return top
# 	}
#
# 	# Tag is name
#     def addBndBox(self,xmin, ymin, width, height, name):
# 	{
#         bndbox = ['xmin'=>xmin, 'ymin'=>ymin, 'xmax'=>(xmin+width), 'ymax'=>(ymin+height)]
#         bndbox['name'] = name
# 		array_push(self._boxlist, bndbox)
# 	}
#
#     def appendObjects(self,top):
# 	{
#
#
#         foreach (self._boxlist as &box)
# 		{
#             object_item = self._domDoc->createElement('object')
# 			object_itemNode = top->appendChild(object_item)
#
# 			name = self._domDoc->createElement('name',  box["name"])
# 			object_itemNode->appendChild(name)
#
# 			pose = self._domDoc->createElement('pose',  Unspecified)
# 			object_itemNode->appendChild(pose)
#
# 			truncated = self._domDoc->createElement('truncated',  "0")
# 			object_itemNode->appendChild(truncated)
#
# 			difficult = self._domDoc->createElement('difficult',  "0")
# 			object_itemNode->appendChild(difficult)
#
# 			bndbox = self._domDoc->createElement('bndbox')
# 			bndboxNode = object_itemNode->appendChild(bndbox)
#
# 			xmin = self._domDoc->createElement('xmin',  box["xmin"])
# 			bndboxNode->appendChild(xmin)
#
# 			ymin = self._domDoc->createElement('ymin',  box["ymin"])
# 			bndboxNode->appendChild(ymin)
#
# 			xmax = self._domDoc->createElement('xmax',  box["xmax"])
# 			bndboxNode->appendChild(xmax)
#
# 			ymax = self._domDoc->createElement('ymax',  box["ymax"])
# 			bndboxNode->appendChild(ymax)
# 		}
# 	}
#
#     def save(self,targetDir):
# 	{
# 		// Generate the XML tree
# 		file = 'file.log'
# 		file_put_contents(file, "Before genXML()\n",FILE_APPEND | LOCK_EX)
#
# 		root = self.genXML()
# 		self.appendObjects(root)
# 		self.prettify()
#
# 		// Replace .jpg by .xml
# 		filename = str_replace(array(".jpg",".JPG"),".xml", self._filename)
# 		fullPath = targetDir. DIRECTORY_SEPARATOR . filename
#
# 		file_put_contents(file, "Save annotations to ". fullPath ."\n",FILE_APPEND | LOCK_EX)
# 		file_put_contents(file, "Xml file: ". filename ."\n",FILE_APPEND | LOCK_EX)
#
# 		self._domDoc->save(fullPath)
# 	}
#
#
# /*data_as_serialize = 'O:8:"stdClass":6:{s:3:"url"s:59:"images/collection/collection_01/famille/20150131_185559.jpg"s:2:"id"s:19:"20150131_185559.jpg"s:6:"folder"s:21:"collection_01/famille"s:5:"width"i:3264s:6:"height"i:2448s:11:"annotations"s:378:"[{"tag":"Anemo probe","x":1618.0064308681672,"y":391.81993569131834,"width":335.84565916398714,"height":374.32797427652736},{"tag":"Anemo probe","x":2552.0771704180065,"y":423.30546623794214,"width":279.87138263665594,"height":279.87138263665594},{"tag":"DND:Drop Nose Device","x":2782.9710610932475,"y":1224.4372990353697,"width":423.30546623794214,"height":1000.540192926045}]"}'
#
# obj = unserialize(data_as_serialize)
#
# folder = obj->{'folder'}
# id     = obj->{'id'}
# width  = obj->{'width'}
# height = obj->{'height'}
# annotations = json_decode(obj->{'annotations'},true)
#
# imageSize = [  "width"  => width ,
# 				"height" => height,
# 				"depth"  => 3 ]
# xml = xmlVocAnnotations(folder, id, imageSize)
#
# foreach (annotations as &annotation)
# 	xml->addBndBox(annotation["x"],
# 					annotation["y"],
# 					annotation["width"],
# 					annotation["height"],
# 					annotation["tag"])
#
# // Write xml to file
# xml->save()*/
#
#
# ?>