<map version="freeplane 1.12.1">
<!--To view this file, download free mind mapping software Freeplane from https://www.freeplane.org -->
<bookmarks>
    <bookmark nodeId="ID_696401721" name="Root" opensAsRoot="true"/>
</bookmarks>
<node TEXT="Automated Classification of Indian Mango Varieties Using Machine Learning and MobileNet-v2 Deep Features" FOLDED="false" ID="ID_696401721" CREATED="1610381621824" MODIFIED="1755797471080" STYLE="oval">
<font SIZE="14"/>
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
<node TEXT="Objective" POSITION="bottom_or_right" ID="ID_372647419" CREATED="1755795989429" MODIFIED="1755796002016" STYLE="bubble">
<edge COLOR="#ff0000"/>
<node TEXT="The study aims to automatically classify Indian mango varieties using lightweight CNNs (MobileNet-v2 and ShuffleNet) combined with traditional machine learning classifiers. The main goal is to create a fast, accurate, and mobile-friendly solution for fruit classification in agriculture and retail sectors." ID="ID_1814583441" CREATED="1755796002020" MODIFIED="1755796022049" STYLE="narrow_hexagon"/>
</node>
<node TEXT="Dataset" POSITION="top_or_left" ID="ID_823125485" CREATED="1755796037838" MODIFIED="1755796045649" STYLE="bubble">
<edge COLOR="#0000ff"/>
<node TEXT="Collected a new dataset of 15 Indian mango varieties (e.g., Alphonso, Langra, Totapuri, Kesar, Dasheri, Himsagar)." ID="ID_675852764" CREATED="1755796047426" MODIFIED="1755796054499"/>
<node TEXT="Total 1,853 images, captured using a smartphone camera under natural daylight with a white background for clarity." ID="ID_1702822648" CREATED="1755796056730" MODIFIED="1755796064206"/>
<node TEXT="Each variety has 100–200 images, resized to 224×224 pixels for CNN input." ID="ID_855649032" CREATED="1755796064513" MODIFIED="1755796071737"/>
<node TEXT="Dataset made public on Mendeley Data (DOI: 10.17632/tk6d98f87d.2)." ID="ID_705170487" CREATED="1755796072629" MODIFIED="1755796078686"/>
</node>
<node TEXT="Methodology" POSITION="bottom_or_right" ID="ID_1668259843" CREATED="1755796096733" MODIFIED="1755796104991" STYLE="bubble">
<edge COLOR="#00ff00"/>
<node TEXT="Feature Extraction (Deep Learning)" ID="ID_1507876639" CREATED="1755796104996" MODIFIED="1755796129261" STYLE="narrow_hexagon">
<node TEXT="Two lightweight CNNs used" ID="ID_667992175" CREATED="1755796133267" MODIFIED="1755796179991" STYLE="bubble">
<node TEXT="MobileNet-v2 (efficient with inverted residuals &amp; depthwise separable convolutions)." ID="ID_1407420075" CREATED="1755796146206" MODIFIED="1755796153336"/>
<node ID="ID_1976525188" CREATED="1755796158233" MODIFIED="1755796158233"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="1105" data-end="1273">
      <li data-start="1203" data-end="1273">
        <p data-start="1205" data-end="1273">
          <strong data-start="1205" data-end="1219">ShuffleNet</strong>&#xa0;(channel shuffling for small-network optimization).
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Extracted deep features from these pretrained networks." ID="ID_1223940517" CREATED="1755796160863" MODIFIED="1755796179985" STYLE="bubble"/>
</node>
<node TEXT="Classification (Machine Learning)" ID="ID_1047140590" CREATED="1755796190377" MODIFIED="1755796198862" STYLE="narrow_hexagon">
<node ID="ID_103782846" CREATED="1755796222248" MODIFIED="1755796222248"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Deep features were classified using <strong data-start="1426" data-end="1461">22 machine learning classifiers</strong>&#xa0;(SVM, KNN, Naïve Bayes, Decision Trees, Ensemble methods).
  </body>
</html>
</richcontent>
</node>
<node TEXT="Best results obtained with Cubic SVM applied to MobileNet-v2 features." ID="ID_500594334" CREATED="1755796236252" MODIFIED="1755796243516"/>
</node>
<node TEXT="Training Setup" ID="ID_240302198" CREATED="1755796294164" MODIFIED="1755796303780" STYLE="narrow_hexagon">
<node TEXT="" ID="ID_786588970" CREATED="1755796303785" MODIFIED="1755796303785">
<node ID="ID_1844691859" CREATED="1755796311248" MODIFIED="1755796311248"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Train/validation/test split: <strong data-start="1664" data-end="1683">70% / 20% / 10%</strong>.
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_1551763910" CREATED="1755796312616" MODIFIED="1755796312616">
<node ID="ID_1636981537" CREATED="1755796320935" MODIFIED="1755796320935"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="1690" data-end="1713">
        <p data-start="1692" data-end="1713">
          Batch size: <strong data-start="1704" data-end="1710">64</strong>.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_438561936" CREATED="1755796322761" MODIFIED="1755796322761">
<node ID="ID_59135411" CREATED="1755796329126" MODIFIED="1755796329126"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="1717" data-end="1746">
        <p data-start="1719" data-end="1746">
          Learning rate: <strong data-start="1734" data-end="1743">0.001</strong>.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_1642239091" CREATED="1755796330365" MODIFIED="1755796330365">
<node ID="ID_1680332516" CREATED="1755796337258" MODIFIED="1755796337258"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="1633" data-end="1833">
      <li data-start="1750" data-end="1833">
        <p data-start="1752" data-end="1833">
          Epochs: <strong data-start="1760" data-end="1769">50–70</strong>&#xa0;(best accuracy at 70 epochs; overfitting observed after 100).
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
</node>
<node TEXT="Results" POSITION="top_or_left" ID="ID_795916520" CREATED="1755796346109" MODIFIED="1755796358019" STYLE="bubble">
<edge COLOR="#ff00ff"/>
<node TEXT="MobileNet-v2 + Cubic SVM:" ID="ID_1020359084" CREATED="1755796358023" MODIFIED="1755796372112" STYLE="oval">
<node TEXT="Validation Accuracy: 99.5%" ID="ID_1418037973" CREATED="1755796377039" MODIFIED="1755796383507"/>
<node TEXT="Test Accuracy: 100%" ID="ID_21409041" CREATED="1755796384760" MODIFIED="1755796390107"/>
<node TEXT="AUC: 1.0" ID="ID_1706283640" CREATED="1755796391085" MODIFIED="1755796396157"/>
<node TEXT="F1 Score: ~99.8%" ID="ID_810421224" CREATED="1755796396366" MODIFIED="1755796401253"/>
</node>
<node TEXT="ShuffleNet + Cubic SVM:" ID="ID_864355668" CREATED="1755796414878" MODIFIED="1755796424764" STYLE="rectangle">
<node ID="ID_522847761" CREATED="1755796431760" MODIFIED="1755796431760"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Accuracy: ~99.4%
  </body>
</html>
</richcontent>
</node>
<node TEXT="AUC: 1.0" ID="ID_532365209" CREATED="1755796441946" MODIFIED="1755796448426"/>
</node>
<node TEXT="MobileNet-v2 consistently outperformed ShuffleNet in both validation and test phases." ID="ID_592515352" CREATED="1755796449198" MODIFIED="1755796454687"/>
</node>
<node TEXT="Strengths" POSITION="bottom_or_right" ID="ID_65664879" CREATED="1755796480456" MODIFIED="1755796493245" STYLE="bubble">
<edge COLOR="#00ffff"/>
<node TEXT="" ID="ID_1501293025" CREATED="1755796493250" MODIFIED="1755796493250">
<node ID="ID_500728925" CREATED="1755796516320" MODIFIED="1755796516320"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="2189" data-end="2497">
      <li data-start="2189" data-end="2254">
        <p data-start="2191" data-end="2254">
          Achieves <strong data-start="2200" data-end="2229">state-of-the-art accuracy</strong>&#xa0;on 15 mango varieties.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
<node ID="ID_608917741" CREATED="1755796516321" MODIFIED="1755796516321"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="2189" data-end="2497">
      <li data-start="2255" data-end="2329">
        <p data-start="2257" data-end="2329">
          Uses <strong data-start="2262" data-end="2282">lightweight CNNs</strong>, making it deployable on <strong data-start="2308" data-end="2326">mobile devices</strong>.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
<node ID="ID_247169831" CREATED="1755796516324" MODIFIED="1755796516324"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="2189" data-end="2497">
      <li data-start="2330" data-end="2391">
        <p data-start="2332" data-end="2391">
          Publicly released <strong data-start="2350" data-end="2361">dataset</strong>&#xa0;encourages reproducibility.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
<node ID="ID_1310555849" CREATED="1755796516325" MODIFIED="1755796516325"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="2189" data-end="2497">
      <li data-start="2392" data-end="2497">
        <p data-start="2394" data-end="2497">
          Demonstrates <strong data-start="2407" data-end="2426">hybrid approach</strong>&#xa0;(deep features + machine learning classifiers) rather than CNN-only.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Limitations" POSITION="top_or_left" ID="ID_590008552" CREATED="1755796524633" MODIFIED="1755796538982" STYLE="bubble">
<edge COLOR="#7c0000"/>
<node TEXT="" ID="ID_1205687775" CREATED="1755796564137" MODIFIED="1755796564137">
<node ID="ID_1254115921" CREATED="1755796575458" MODIFIED="1755796575458"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="2524" data-end="2615">
        <p data-start="2526" data-end="2615">
          Dataset relatively <strong data-start="2545" data-end="2569">small (1,853 images)</strong>&#xa0;compared to other deep learning benchmarks.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_916890610" CREATED="1755796576950" MODIFIED="1755796576950">
<node ID="ID_498314412" CREATED="1755796587613" MODIFIED="1755796587613"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="2616" data-end="2774">
        <p data-start="2618" data-end="2774">
          Images captured under <strong data-start="2640" data-end="2663">controlled settings</strong>&#xa0;(white background, fixed distance), so generalization to <strong data-start="2721" data-end="2758">real-world farm/market conditions</strong>&#xa0;is uncertain.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_724201254" CREATED="1755796600510" MODIFIED="1755796600510">
<node ID="ID_711418970" CREATED="1755796604828" MODIFIED="1755796604828"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="2775" data-end="2857">
        <p data-start="2777" data-end="2857">
          Focuses only on <strong data-start="2793" data-end="2819">variety classification</strong>, not ripeness or disease detection.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Does not benchmark against modern end-to-end deep CNNs (e.g., EfficientNet, ViT, YOLO)." ID="ID_791964822" CREATED="1755796611916" MODIFIED="1755796613853"/>
</node>
<node TEXT="Future Imporovement" POSITION="bottom_or_right" ID="ID_1121958103" CREATED="1755796877243" MODIFIED="1755796891897" STYLE="bubble">
<edge COLOR="#00007c"/>
<node TEXT="" ID="ID_1209445876" CREATED="1755796891902" MODIFIED="1755796891902">
<node ID="ID_223318974" CREATED="1755796918924" MODIFIED="1755796918924"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="3916" data-end="3967">
        <p data-start="3918" data-end="3967">
          <strong data-start="3918" data-end="3927">Data:</strong>&#xa0;Expand, diversify, cross-domain test.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_1598127666" CREATED="1755796927054" MODIFIED="1755796927054">
<node ID="ID_1024579198" CREATED="1755796934903" MODIFIED="1755796934903"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="3968" data-end="4041">
        <p data-start="3970" data-end="4041">
          <strong data-start="3970" data-end="3981">Models:</strong>&#xa0;Go beyond MobileNet/ShuffleNet → EfficientNet, ViT, YOLO.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_717659871" CREATED="1755796939399" MODIFIED="1755796939399">
<node ID="ID_219398768" CREATED="1755796943272" MODIFIED="1755796943272"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="4042" data-end="4110">
        <p data-start="4044" data-end="4110">
          <strong data-start="4044" data-end="4056">Metrics:</strong>&#xa0;Add macro-F1, confusion per class, latency, memory.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node ID="ID_1394682148" CREATED="1755796948149" MODIFIED="1755796997326"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <span style="font-weight: bold;">Deployment</span>: Optimize for mobile, test real-time performance.
    </p>
  </body>
</html>
</richcontent>
<font BOLD="false"/>
</node>
<node ID="ID_1632492763" CREATED="1755796954456" MODIFIED="1755797000791"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <span style="font-weight: bold;">Explainability</span>: Grad-CAM visualizations.
    </p>
  </body>
</html>
</richcontent>
</node>
<node ID="ID_1533424205" CREATED="1755796960505" MODIFIED="1755797003340"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      <span style="font-weight: bold;">Extension</span>: Ripeness, disease, multi-task &amp; multi-modal classification.
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Future Research Directions" POSITION="top_or_left" ID="ID_1105951649" CREATED="1755797072087" MODIFIED="1755797086854" STYLE="bubble">
<edge COLOR="#007c00"/>
<node TEXT="Extend to ripeness grading &amp; disease detection (combine with the methods from your second paper)." ID="ID_1751111379" CREATED="1755797086859" MODIFIED="1755797106987"/>
<node TEXT="Explore multi-modal features: combine image + weight/size/color sensor data for better classification." ID="ID_523244962" CREATED="1755797110771" MODIFIED="1755797112485"/>
<node TEXT="Investigate active learning: system improves as farmers upload mislabeled/misclassified examples." ID="ID_1859489698" CREATED="1755797119288" MODIFIED="1755797120978"/>
<node TEXT="Move towards automated grading systems for export quality assurance (size, defects, ripeness, variety)." ID="ID_1604870439" CREATED="1755797127818" MODIFIED="1755797129320"/>
</node>
</node>
</map>
