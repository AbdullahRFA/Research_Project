<map version="freeplane 1.12.1">
<!--To view this file, download free mind mapping software Freeplane from https://www.freeplane.org -->
<bookmarks>
    <bookmark nodeId="ID_696401721" name="Root" opensAsRoot="true"/>
</bookmarks>
<node TEXT="CNN-BASED SOLUTION FOR MANGO CLASSIFICATION IN&#xa;AGRICULTURAL ENVIRONMENTS" FOLDED="false" ID="ID_696401721" CREATED="1610381621824" MODIFIED="1755793942082" STYLE="oval">
<font SIZE="18"/>
<hook NAME="MapStyle" zoom="0.5">
    <properties edgeColorConfiguration="#808080ff,#ff0000ff,#0000ffff,#00ff00ff,#ff00ffff,#00ffffff,#7c0000ff,#00007cff,#007c00ff,#7c007cff,#007c7cff,#7c7c00ff" show_icon_for_attributes="true" auto_compact_layout="true" show_tags="UNDER_NODES" associatedTemplateLocation="template:/standard-1.6.mm" show_note_icons="true" fit_to_viewport="false" show_icons="BESIDE_NODES" showTagCategories="false"/>
    <tags category_separator="::"/>

<map_styles>
<stylenode LOCALIZED_TEXT="styles.root_node" STYLE="oval" UNIFORM_SHAPE="true" VGAP_QUANTITY="24 pt">
<font SIZE="24"/>
<stylenode LOCALIZED_TEXT="styles.predefined" POSITION="bottom_or_right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="default" ID="ID_271890427" ICON_SIZE="12 pt" COLOR="#000000" STYLE="fork">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="200" DASH="" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_271890427" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<font NAME="SansSerif" SIZE="10" BOLD="false" ITALIC="false"/>
<richcontent TYPE="DETAILS" CONTENT-TYPE="plain/auto"/>
<richcontent TYPE="NOTE" CONTENT-TYPE="plain/auto"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.details"/>
<stylenode LOCALIZED_TEXT="defaultstyle.tags">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.attributes">
<font SIZE="9"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.note" COLOR="#000000" BACKGROUND_COLOR="#ffffff" TEXT_ALIGN="LEFT"/>
<stylenode LOCALIZED_TEXT="defaultstyle.floating">
<edge STYLE="hide_edge"/>
<cloud COLOR="#f0f0f0" SHAPE="ROUND_RECT"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.selection" BACKGROUND_COLOR="#afd3f7" BORDER_COLOR_LIKE_EDGE="false" BORDER_COLOR="#afd3f7"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.user-defined" POSITION="bottom_or_right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="styles.topic" COLOR="#18898b" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subtopic" COLOR="#cc3300" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subsubtopic" COLOR="#669900">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.important" ID="ID_67550811">
<icon BUILTIN="yes"/>
<arrowlink COLOR="#003399" TRANSPARENCY="255" DESTINATION="ID_67550811"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.flower" COLOR="#ffffff" BACKGROUND_COLOR="#255aba" STYLE="oval" TEXT_ALIGN="CENTER" BORDER_WIDTH_LIKE_EDGE="false" BORDER_WIDTH="22 pt" BORDER_COLOR_LIKE_EDGE="false" BORDER_COLOR="#f9d71c" BORDER_DASH_LIKE_EDGE="false" BORDER_DASH="CLOSE_DOTS" MAX_WIDTH="6 cm" MIN_WIDTH="3 cm"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.AutomaticLayout" POSITION="bottom_or_right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="AutomaticLayout.level.root" COLOR="#000000" STYLE="oval" SHAPE_HORIZONTAL_MARGIN="10 pt" SHAPE_VERTICAL_MARGIN="10 pt">
<font SIZE="18"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,1" COLOR="#0033ff">
<font SIZE="16"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,2" COLOR="#00b439">
<font SIZE="14"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,3" COLOR="#990000">
<font SIZE="12"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,4" COLOR="#111111">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,5"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,6"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,7"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,8"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,9"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,10"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,11"/>
</stylenode>
</stylenode>
</map_styles>
</hook>
<hook NAME="AutomaticEdgeColor" COUNTER="8" RULE="ON_BRANCH_CREATION"/>
<node TEXT="Methology" POSITION="bottom_or_right" ID="ID_536964738" CREATED="1755794369335" MODIFIED="1755794521766" STYLE="bubble">
<edge COLOR="#ff0000"/>
<node TEXT="Dataset Collection" ID="ID_331952513" CREATED="1755794523807" MODIFIED="1755794550365" STYLE="bubble">
<node TEXT="" ID="ID_240695884" CREATED="1755794554084" MODIFIED="1755794554084">
<node TEXT="~17,000 images across 3 classes (raw, ripe, bad).&#xa;Augmented with flipping, rotation, and blurring." ID="ID_43895112" CREATED="1755794568297" MODIFIED="1755795431836"/>
</node>
<node TEXT="" ID="ID_799513089" CREATED="1755794569483" MODIFIED="1755794569483">
<node TEXT="~4,000 images across 5 classes (alternaria, anthracnose, black mold rot, stem end rot, healthy).&#xa;&#xa;Balanced per class (≈300 training samples each)." ID="ID_1690263073" CREATED="1755794579815" MODIFIED="1755795448302"/>
</node>
<node TEXT="" ID="ID_1419755860" CREATED="1755794581103" MODIFIED="1755794581103">
<node TEXT="~11,000 images containing mango bounding boxes (mixed fruits/vegetables dataset).&#xa;&#xa;Used for training object detectors." ID="ID_489761904" CREATED="1755794591330" MODIFIED="1755795459049"/>
</node>
</node>
<node TEXT="Preprocessing" ID="ID_223943892" CREATED="1755794598562" MODIFIED="1755794613744" STYLE="bubble">
<node TEXT="Resize images (224x224 or 227x227)" ID="ID_486329213" CREATED="1755794613749" MODIFIED="1755794625608"/>
<node TEXT="Data augmentation (flip, rotation, blur, crop)" ID="ID_458124101" CREATED="1755794625878" MODIFIED="1755794633720"/>
<node TEXT="Split into train / validation / test sets" ID="ID_1535499906" CREATED="1755794633977" MODIFIED="1755794647758"/>
</node>
<node TEXT="Object Detection" ID="ID_409671741" CREATED="1755794649224" MODIFIED="1755794666499" STYLE="bubble">
<node TEXT="" ID="ID_1685463870" CREATED="1755794657562" MODIFIED="1755794657562">
<node ID="ID_93057276" CREATED="1755795539757" MODIFIED="1755795539757"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="1141" data-end="1219">
        <p data-start="1143" data-end="1219">
          Compared <strong data-start="1152" data-end="1189">R-CNN (AlexNet, VGG-16 backbones)</strong>&#xa0;and a <strong data-start="1196" data-end="1216">Cascade detector</strong>.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
<node ID="ID_175272525" CREATED="1755795539758" MODIFIED="1755795539758"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="1222" data-end="1306">
        <p data-start="1224" data-end="1306">
          Cascade chosen for final deployment due to faster speed and acceptable accuracy.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Mango Region Extraction" ID="ID_23571916" CREATED="1755794701955" MODIFIED="1755794710288" STYLE="bubble"/>
<node TEXT="Classification (ResNet-18)" ID="ID_861757568" CREATED="1755794739271" MODIFIED="1755794749596" STYLE="bubble">
<node TEXT="" ID="ID_1526085264" CREATED="1755794749601" MODIFIED="1755794749601">
<node ID="ID_1545211820" CREATED="1755795573300" MODIFIED="1755795573300"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="1342" data-end="1602">
      <li data-start="1342" data-end="1419">
        <p data-start="1344" data-end="1419">
          <strong data-start="1344" data-end="1357">ResNet-18</strong>&#xa0;used for both ripeness and disease tasks (separate models).
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
<node ID="ID_803332576" CREATED="1755795573301" MODIFIED="1755795573301"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="1342" data-end="1602">
      <li data-start="1422" data-end="1507">
        <p data-start="1424" data-end="1507">
          Modified final layers for 3-class (ripeness) or 5-class (disease) softmax output.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
<node ID="ID_409851296" CREATED="1755795573304" MODIFIED="1755795573304"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="1342" data-end="1602">
      <li data-start="1510" data-end="1602">
        <p data-start="1512" data-end="1602">
          Transfer learning with ImageNet weights; trained for <strong data-start="1565" data-end="1578">10 epochs</strong>&#xa0;&#xa0;using SGDM optimizer.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Training" ID="ID_352839011" CREATED="1755794777814" MODIFIED="1755794785418" STYLE="bubble">
<node TEXT="Optimizer: SGDM" ID="ID_121180295" CREATED="1755794785423" MODIFIED="1755794793884"/>
<node TEXT="Batch size: 32" ID="ID_1092889411" CREATED="1755794796772" MODIFIED="1755794804221"/>
<node TEXT="Learning rate (Ripeness: 1e-3, Disease: 1e-2)" ID="ID_1873402931" CREATED="1755794804377" MODIFIED="1755794812353"/>
<node TEXT="Epochs: 10" ID="ID_1103094876" CREATED="1755794828598" MODIFIED="1755794830683"/>
</node>
<node TEXT="Evaluation" ID="ID_1387562762" CREATED="1755794838271" MODIFIED="1755794847133" STYLE="bubble">
<node TEXT="Accuracy" ID="ID_1099817299" CREATED="1755794847138" MODIFIED="1755794855489"/>
<node TEXT="Per-class accuracy" ID="ID_9395887" CREATED="1755794855883" MODIFIED="1755794866208"/>
<node TEXT="onfusion matrix" ID="ID_451292594" CREATED="1755794866497" MODIFIED="1755794872788"/>
<node TEXT="(Detection: bounding box quality, false positives)" ID="ID_1248745146" CREATED="1755794873333" MODIFIED="1755794895150"/>
</node>
<node TEXT="Deployment" ID="ID_1461930294" CREATED="1755794908736" MODIFIED="1755794922094" STYLE="oval">
<node TEXT="" ID="ID_766612715" CREATED="1755794922098" MODIFIED="1755794922098">
<node TEXT="GUI built in MATLAB App Designer" ID="ID_781992434" CREATED="1755794939677" MODIFIED="1755794939677"/>
</node>
<node TEXT="Input: Image / Live Camera" ID="ID_1967059999" CREATED="1755794941510" MODIFIED="1755794948141"/>
<node TEXT="Output: Mango ripeness or disease result" ID="ID_142486844" CREATED="1755794949225" MODIFIED="1755794957616"/>
</node>
</node>
<node TEXT="" POSITION="top_or_left" ID="ID_1310208984" CREATED="1755795478431" MODIFIED="1755795478433">
<edge COLOR="#ff00ff"/>
<node TEXT="Results" ID="ID_136366667" CREATED="1755795591165" MODIFIED="1755795603266" STYLE="bubble">
<node TEXT="Ripeness classification (ResNet-18): ~89.5% accuracy; high per-class performance (bad: 95%, ripe: 93%, raw: 89%)." ID="ID_1627077021" CREATED="1755795604173" MODIFIED="1755795614684"/>
<node TEXT="Disease classification (ResNet-18): ≥88% accuracy across classes; healthy class achieved 100% accuracy." ID="ID_1970974770" CREATED="1755795618761" MODIFIED="1755795629775"/>
<node TEXT="Detection: R-CNN gave decent bounding boxes, but cascade detector was more efficient, making it better for practical use." ID="ID_964194215" CREATED="1755795630046" MODIFIED="1755795636990"/>
<node TEXT="Combined model (ripeness + disease in one net): lower precision and confusion; separate models recommended." ID="ID_26364205" CREATED="1755795637230" MODIFIED="1755795643851"/>
</node>
</node>
<node TEXT="Strengths" POSITION="bottom_or_right" ID="ID_633086036" CREATED="1755795692769" MODIFIED="1755795702678" STYLE="oval">
<edge COLOR="#7c0000"/>
<node TEXT="End-to-end workflow (detection → classification → GUI)." ID="ID_1826961796" CREATED="1755795702685" MODIFIED="1755795711391"/>
<node TEXT="Transfer learning reduces training cost and data requirement." ID="ID_1824965840" CREATED="1755795711714" MODIFIED="1755795717460"/>
<node TEXT="" ID="ID_695611614" CREATED="1755795719938" MODIFIED="1755795719938">
<node ID="ID_1072577907" CREATED="1755795731293" MODIFIED="1755795731293"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Practical deployment tool (user-friendly GUI).
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_1395830244" CREATED="1755795733080" MODIFIED="1755795733080">
<node ID="ID_454304435" CREATED="1755795744247" MODIFIED="1755795744247"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Balanced datasets and preprocessing steps well documented.
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Limitation" POSITION="top_or_left" ID="ID_316981812" CREATED="1755795759511" MODIFIED="1755795778361" STYLE="wide_hexagon">
<edge COLOR="#00007c"/>
<node TEXT="Only 10 epochs training — deeper models (ResNet-50/101) underperformed due to under-training." ID="ID_259630465" CREATED="1755795778366" MODIFIED="1755795787184"/>
<node TEXT="" ID="ID_1045705091" CREATED="1755795788039" MODIFIED="1755795788039">
<node ID="ID_532143946" CREATED="1755795797997" MODIFIED="1755795797997"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="2641" data-end="2731">
        <p data-start="2643" data-end="2731">
          No standardized metrics like <strong data-start="2672" data-end="2679">mAP</strong>&#xa0;for detection or <strong data-start="2697" data-end="2709">macro-F1</strong>&#xa0;&#xa0;for classification.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_1150204669" CREATED="1755795799447" MODIFIED="1755795799447">
<node ID="ID_417298619" CREATED="1755795806891" MODIFIED="1755795806891"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="2732" data-end="2832">
        <p data-start="2734" data-end="2832">
          Evaluations limited to controlled datasets; <strong data-start="2778" data-end="2800">real-world testing</strong>&#xa0;&#xa0;(farms/markets) not included.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_1308780829" CREATED="1755795808183" MODIFIED="1755795808183">
<node ID="ID_1761635900" CREATED="1755795818681" MODIFIED="1755795818681"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Cascade detector chosen mainly for speed but may underperform compared to modern detectors (YOLO, Faster R-CNN).
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Objective" POSITION="bottom_or_right" ID="ID_1163997680" CREATED="1755795889847" MODIFIED="1755795910381" STYLE="narrow_hexagon">
<edge COLOR="#007c00"/>
<node TEXT="The paper proposes an automated mango analysis system that can detect mangoes in images and then classify them by ripeness stage (raw, ripe, bad) and by disease category (five common diseases). The goal is to reduce manual, subjective inspection and build a system suitable for agriculture, quality control, and export industries." ID="ID_1660652559" CREATED="1755795910386" MODIFIED="1755795932066" STYLE="bubble"/>
</node>
</node>
</map>
