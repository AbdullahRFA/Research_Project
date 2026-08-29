<map version="freeplane 1.12.1">
<!--To view this file, download free mind mapping software Freeplane from https://www.freeplane.org -->
<bookmarks>
    <bookmark nodeId="ID_696401721" name="Root" opensAsRoot="true"/>
</bookmarks>
<node TEXT="An Application of Machine Learning Methods to  Detect Mango Varieties" FOLDED="false" ID="ID_696401721" CREATED="1610381621824" MODIFIED="1755792769933" STYLE="oval">
<font SIZE="14"/>
<hook NAME="MapStyle">
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
<hook NAME="AutomaticEdgeColor" COUNTER="5" RULE="ON_BRANCH_CREATION"/>
<node TEXT="What the paper did (in one minute)" POSITION="bottom_or_right" ID="ID_1825019659" CREATED="1755783591948" MODIFIED="1755783700002" STYLE="bubble">
<edge COLOR="#ff0000"/>
<node TEXT="Task" ID="ID_348831913" CREATED="1755783781622" MODIFIED="1755783804677" STYLE="bubble">
<node TEXT="Classify mango varieties from images" ID="ID_824589809" CREATED="1755783807964" MODIFIED="1755783814277"/>
</node>
<node TEXT="Data" ID="ID_1099385343" CREATED="1755783834812" MODIFIED="1755783855313" STYLE="bubble">
<node TEXT="“Mango Variety” dataset (1,661 images, 15 classes; controlled capture) and simple augmentations (flips/90° rotations)." ID="ID_1889703820" CREATED="1755783867987" MODIFIED="1755783870514"/>
</node>
<node TEXT="Models compared" ID="ID_656664526" CREATED="1755783914736" MODIFIED="1755783926027" STYLE="bubble">
<node TEXT="MobileNetV2, Xception, VGG16, ResNet50V2 (transfer learning; 10 epochs)." LOCALIZED_STYLE_REF="styles.subtopic" ID="ID_966982701" CREATED="1755783928950" MODIFIED="1755792598204" STYLE="bubble">
<node TEXT="" ID="ID_1956162937" CREATED="1755792266787" MODIFIED="1755792266787">
<node TEXT="MobileNetV2" ID="ID_624843539" CREATED="1755792274318" MODIFIED="1755792512685" STYLE="bubble">
<node TEXT="" ID="ID_1345521087" CREATED="1755792281401" MODIFIED="1755792281401">
<node ID="ID_752437074" CREATED="1755792293767" MODIFIED="1755792526434" STYLE="bubble"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Core idea
  </body>
</html>
</richcontent>
<node TEXT="Lightweight CNN designed for mobile/edge devices." ID="ID_1689174273" CREATED="1755792303644" MODIFIED="1755792306761"/>
</node>
</node>
<node TEXT="How it works here" ID="ID_1559834482" CREATED="1755792307673" MODIFIED="1755792526438" STYLE="bubble">
<node TEXT="Uses depthwise separable convolutions (splitting spatial and channel-wise filtering) to reduce computation.&#xa;&#xa;Introduces inverted residual blocks with linear bottlenecks: instead of expanding then compressing, it keeps most layers narrow, saving memory.&#xa;&#xa;For mango classification, this helps achieve good accuracy with small model size (only ~2.2M parameters) but the paper shows unusually high inference time due to unclear setup." ID="ID_125301533" CREATED="1755792317991" MODIFIED="1755792325983"/>
</node>
</node>
</node>
<node TEXT="Xception" ID="ID_532333307" CREATED="1755792332555" MODIFIED="1755792512691" STYLE="bubble">
<node TEXT="Core idea" ID="ID_1708064799" CREATED="1755792339472" MODIFIED="1755792512693" STYLE="bubble">
<node TEXT="“Extreme Inception” — replaces Inception modules with depthwise separable convolutions everywhere." ID="ID_525274845" CREATED="1755792350160" MODIFIED="1755792367448"/>
</node>
<node TEXT="How it works here" ID="ID_239741513" CREATED="1755792367849" MODIFIED="1755792512693" STYLE="bubble">
<node TEXT="Each convolution is factorized into a channel-wise convolution + pointwise (1×1) convolution, reducing computation while keeping strong feature extraction.&#xa;&#xa;Very effective at capturing fine-grained textures and shapes of mango skins.&#xa;&#xa;In the paper, it gives the best results (≈99.4% accuracy, precision, F1, recall), showing it learns discriminative patterns between similar-looking mango varieties." ID="ID_1265079473" CREATED="1755792370537" MODIFIED="1755792379230"/>
</node>
</node>
<node TEXT="VGG16" ID="ID_105303302" CREATED="1755792380507" MODIFIED="1755792512696" STYLE="bubble">
<node TEXT="Core idea" ID="ID_174389636" CREATED="1755792387992" MODIFIED="1755792512695" STYLE="bubble">
<node TEXT="Deep CNN with stacked 3×3 convolutions and max-pooling, simple but heavy." ID="ID_78383584" CREATED="1755792395253" MODIFIED="1755792404763"/>
</node>
<node TEXT="How it works here" ID="ID_1770209066" CREATED="1755792405163" MODIFIED="1755792512697" STYLE="bubble">
<node TEXT="Extracts hierarchical features (edges → textures → object parts) with a straightforward pipeline.&#xa;&#xa;Contains ~138M parameters, making it large and computationally expensive.&#xa;&#xa;In this paper, its performance looks inconsistent (accuracy ~99% but macro-F1 only ~83%), likely due to class imbalance sensitivity and insufficient training (only 10 epochs)." ID="ID_1351436566" CREATED="1755792412722" MODIFIED="1755792432719"/>
</node>
</node>
<node TEXT="ResNet50V2" ID="ID_296995064" CREATED="1755792434676" MODIFIED="1755792554392" STYLE="bubble">
<node TEXT="Core idea" ID="ID_972960943" CREATED="1755792439260" MODIFIED="1755792554388" STYLE="bubble">
<node TEXT="Residual Networks — solves vanishing gradient by adding skip connections (shortcut paths) so gradients can flow more easily." ID="ID_160155210" CREATED="1755792445355" MODIFIED="1755792451260"/>
</node>
<node TEXT="" ID="ID_722537010" CREATED="1755792456457" MODIFIED="1755792456457"/>
<node ID="ID_715074162" CREATED="1755792463716" MODIFIED="1755792554394" STYLE="bubble"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    How it works here
  </body>
</html>
</richcontent>
<node TEXT="ResNet50V2 refines the original with batch normalization and activation order adjustments.&#xa;&#xa;Helps train deeper networks efficiently, capturing complex mango features.&#xa;&#xa;In this paper, it performs better than VGG16 but lower than Xception and MobileNetV2, suggesting it needed more epochs or better tuning." ID="ID_674215706" CREATED="1755792481329" MODIFIED="1755792491885"/>
</node>
</node>
</node>
</node>
<node TEXT="Headline result" ID="ID_1102851873" CREATED="1755783959512" MODIFIED="1755783984743" STYLE="bubble">
<node TEXT="Xception is reported best (Accuracy 99.43%, Precision 99.47%, F1 99.43%, Recall 99.48%); MobileNetV2 next; ResNet50V2 third; VGG16 lags." ID="ID_1145790783" CREATED="1755783963317" MODIFIED="1755783973560"/>
</node>
<node TEXT="Claimed novelty" ID="ID_1693370902" CREATED="1755784014615" MODIFIED="1755784024993" STYLE="bubble">
<node TEXT="Outperforms several prior works on mango classification/grading" ID="ID_19156306" CREATED="1755784026705" MODIFIED="1755784036310"/>
</node>
</node>
<node TEXT="Strengths" POSITION="top_or_left" ID="ID_1594079632" CREATED="1755784379774" MODIFIED="1755784442848" VSHIFT_QUANTITY="0.96644 pt" STYLE="bubble">
<edge COLOR="#0000ff"/>
<node TEXT="Clear comparative baseline across four popular CNN families with standard metrics (Accuracy/Precision/Recall/F1, MCC, MSE" ID="ID_992747892" CREATED="1755784387732" MODIFIED="1755784406109"/>
<node TEXT="" ID="ID_1301367005" CREATED="1755784449518" MODIFIED="1755784497660" HGAP_QUANTITY="8.68456 pt">
<node ID="ID_608074062" CREATED="1755784476181" MODIFIED="1755784476181"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <strong data-start="1044" data-end="1069">Usable public dataset</strong>&#xa0;and a straightforward pipeline (preprocess → augment → split → train → evaluate).
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Practical motivation for sorting/grading and marketing pipelines" ID="ID_1765033948" CREATED="1755784516656" MODIFIED="1755784518642"/>
</node>
<node TEXT="Issues &amp; gaps (opportunities for my research)" POSITION="bottom_or_right" ID="ID_1620786375" CREATED="1755784554748" MODIFIED="1755784581814" STYLE="narrow_hexagon">
<edge COLOR="#00ff00"/>
<node TEXT="Dataset limitations / domain shift" ID="ID_1488666352" CREATED="1755784626846" MODIFIED="1755792965842" STYLE="bubble">
<node TEXT="" ID="ID_1049102129" CREATED="1755784727403" MODIFIED="1755784727403">
<node ID="ID_407979944" CREATED="1755784751863" MODIFIED="1755784751863"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Images are from a <em data-start="1418" data-end="1430">controlled</em>&#xa0;environment; generalization to “in-the-wild” (on-tree, markets, variable lighting/backgrounds/occlusions) is untested. This matters for real deployment in Bangladesh markets.
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Metric inconsistencies" ID="ID_476002249" CREATED="1755784641033" MODIFIED="1755792965841" STYLE="bubble">
<node TEXT="Table shows VGG16 with Accuracy ≈ 99.15% but Precision/F1 ≈ 83%—that mismatch signals either a reporting error, class imbalance effects, or a non-macro averaging choice that isn’t explained.&#xa;&#xa;Inference time numbers conflict with expectations: MobileNetV2 is listed at 7255 ms while Xception is 200 ms (MobileNet is usually faster/smaller). The measurement setup (hardware/batch size/FP32 vs. int8) is not described, so these timings are not comparable yet." ID="ID_471577265" CREATED="1755784766183" MODIFIED="1755784769496"/>
</node>
<node TEXT="Experimental detail missing for reproducibility" ID="ID_1662846797" CREATED="1755784654528" MODIFIED="1755792965840" STYLE="bubble">
<node TEXT="" ID="ID_1582668153" CREATED="1755784772912" MODIFIED="1755784772912">
<node ID="ID_157393867" CREATED="1755784792680" MODIFIED="1755784792680"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    No precise train/val/test split proportions, no random seed handling or subject-level grouping (risk of <strong data-start="2410" data-end="2426">data leakage</strong>&#xa0;if different photos of the same fruit end up in train and test), limited augmentation description, and no full hyperparameter table (optimizer schedule, lr, wd, early stopping).
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="" ID="ID_714594964" CREATED="1755784665960" MODIFIED="1755784665960">
<node ID="ID_1895197759" CREATED="1755784705481" MODIFIED="1755792965830" STYLE="bubble"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Comparisons vs. prior work
  </body>
</html>
</richcontent>
<node TEXT="The paper’s comparison table aggregates works with different tasks/datasets (quality grading vs. variety ID; on-tree vs. studio), so the “our method is higher” conclusion isn’t apples-to-apples. You can clean this up with standardized cross-dataset tests" ID="ID_1187591304" CREATED="1755784830533" MODIFIED="1755784832775"/>
</node>
</node>
<node TEXT="Only 10 epochs" ID="ID_199606542" CREATED="1755784710687" MODIFIED="1755792965818" STYLE="bubble">
<node TEXT="Fixed short training may handicap some models (e.g., VGG/ResNet need different schedules/regularization); fairness across architectures is uncertain." ID="ID_1696068204" CREATED="1755784802634" MODIFIED="1755784822088"/>
</node>
</node>
<node TEXT="Extensions" POSITION="top_or_left" ID="ID_1684507363" CREATED="1755785086828" MODIFIED="1755785113939" STYLE="narrow_hexagon">
<edge COLOR="#ff00ff"/>
<node TEXT="Joint tasks" ID="ID_1614091503" CREATED="1755785137455" MODIFIED="1755785272926" STYLE="bubble">
<node TEXT="" ID="ID_990416810" CREATED="1755785150691" MODIFIED="1755785150691">
<node ID="ID_692505705" CREATED="1755785159503" MODIFIED="1755785159503"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Variety <strong data-start="6208" data-end="6213">+</strong>&#xa0;ripeness/grade detection; multi-task heads
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="Multi-modal fusion" ID="ID_334713005" CREATED="1755785168423" MODIFIED="1755785272946" STYLE="bubble">
<node TEXT="Simple weight-only ViT vision + weight/size meta-features from a low-cost scale/ruler." ID="ID_856257440" CREATED="1755785178288" MODIFIED="1755785192589"/>
</node>
<node TEXT="Low-light assistance" ID="ID_326560295" CREATED="1755785196819" MODIFIED="1755785272951" STYLE="bubble">
<node TEXT="Train with synthetic low-light augmentation and test with a tiny denoiser pre-pass." ID="ID_99334769" CREATED="1755785205814" MODIFIED="1755785224540"/>
</node>
<node TEXT="Fairness &amp; bias" ID="ID_708800347" CREATED="1755785233382" MODIFIED="1755785272951" STYLE="bubble">
<node TEXT="Analyze per-variety and per-capture-condition disparities; report mitigations." ID="ID_519802512" CREATED="1755785236134" MODIFIED="1755785248081"/>
</node>
</node>
<node TEXT="Methology" POSITION="bottom_or_right" ID="ID_1558203889" CREATED="1755792786651" MODIFIED="1755792824137" STYLE="wide_hexagon">
<edge COLOR="#00ffff"/>
<node TEXT="" ID="ID_1649288272" CREATED="1755792829443" MODIFIED="1755792829443">
<node ID="ID_401721560" CREATED="1755792840661" MODIFIED="1755792937748" STYLE="bubble"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    Dataset
  </body>
</html>
</richcontent>
<node TEXT="" ID="ID_1909647370" CREATED="1755792845369" MODIFIED="1755792845369">
<node ID="ID_1188301220" CREATED="1755792864694" MODIFIED="1755792864694"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="170" data-end="261">
        <p data-start="172" data-end="261">
          Used the <strong data-start="181" data-end="206">Mango Variety dataset</strong>&#xa0;with <strong data-start="212" data-end="228">1,661 images</strong>&#xa0;&#xa0;of <strong data-start="232" data-end="258">15 different varieties</strong>.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_651022419" CREATED="1755792855449" MODIFIED="1755792855449">
<node ID="ID_1882983931" CREATED="1755792875243" MODIFIED="1755792875243"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="265" data-end="349">
        <p data-start="267" data-end="349">
          Images captured in a <strong data-start="288" data-end="314">controlled environment</strong>&#xa0;(uniform background &amp; lighting).
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Preprocessing: resized to fixed dimensions for model input." ID="ID_694579618" CREATED="1755792883955" MODIFIED="1755792887720"/>
<node TEXT="" ID="ID_1297272538" CREATED="1755792890112" MODIFIED="1755792890112">
<node ID="ID_36349601" CREATED="1755792899229" MODIFIED="1755792899229"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="170" data-end="506">
      <li data-start="424" data-end="506">
        <p data-start="426" data-end="506">
          <strong data-start="426" data-end="443">Augmentation:</strong>&#xa0;applied flipping and 90° rotations to increase data variety.
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
<node TEXT="" ID="ID_1571126632" CREATED="1755792994429" MODIFIED="1755792994429">
<node ID="ID_824000990" CREATED="1755793001745" MODIFIED="1755793021412" STYLE="bubble"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <p data-start="511" data-end="531">
      <strong data-start="511" data-end="529">Data Splitting</strong>
    </p>
  </body>
</html>
</richcontent>
<node TEXT="" ID="ID_1652809838" CREATED="1755793021417" MODIFIED="1755793021417">
<node ID="ID_1268788149" CREATED="1755793029340" MODIFIED="1755793029340"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="535" data-end="623">
        <p data-start="537" data-end="623">
          Dataset divided into <strong data-start="558" data-end="587">training and testing sets</strong>&#xa0;(exact ratio not clearly stated).
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="Used split to train models and evaluate classification performance." ID="ID_1787420909" CREATED="1755793031475" MODIFIED="1755793038117"/>
</node>
</node>
<node TEXT="Models Used" ID="ID_1657451947" CREATED="1755793047450" MODIFIED="1755793059078" STYLE="bubble">
<node TEXT="MobileNetV2 (lightweight CNN for mobile deployment)." ID="ID_25943065" CREATED="1755793059082" MODIFIED="1755793068157"/>
<node TEXT="Xception (depthwise separable convolutions, high accuracy)." ID="ID_52686438" CREATED="1755793068495" MODIFIED="1755793073060"/>
<node TEXT="" ID="ID_805916373" CREATED="1755793076171" MODIFIED="1755793076171">
<node ID="ID_1650660070" CREATED="1755793089141" MODIFIED="1755793089141"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="859" data-end="922">
        <p data-start="861" data-end="922">
          <strong data-start="861" data-end="870">VGG16</strong>&#xa0;(classic deep CNN with stacked 3×3 convolutions).
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_416024480" CREATED="1755793091802" MODIFIED="1755793091802">
<node ID="ID_1163458689" CREATED="1755793100380" MODIFIED="1755793100380"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="724" data-end="998">
      <li data-start="926" data-end="998">
        <p data-start="928" data-end="998">
          <strong data-start="928" data-end="942">ResNet50V2</strong>&#xa0;(residual skip connections for deep stable training).
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
</node>
<node TEXT="" ID="ID_868377150" CREATED="1755793109496" MODIFIED="1755793109496">
<node TEXT="Training Setup" ID="ID_1348888642" CREATED="1755793121212" MODIFIED="1755793130904" STYLE="bubble">
<node TEXT="Transfer learning: models initialized with ImageNet weights." ID="ID_1386350752" CREATED="1755793130910" MODIFIED="1755793136520"/>
<node TEXT="" ID="ID_360970796" CREATED="1755793136856" MODIFIED="1755793136856">
<node ID="ID_792196293" CREATED="1755793141199" MODIFIED="1755793141199"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul>
      <li data-start="1099" data-end="1135">
        <p data-start="1101" data-end="1135">
          Fine-tuned on the mango dataset.
        </p>
      </li>
    </ul>
  </body>
</html>
</richcontent>
</node>
</node>
<node TEXT="" ID="ID_1357565738" CREATED="1755793142515" MODIFIED="1755793142515">
<node ID="ID_1151416159" CREATED="1755793151678" MODIFIED="1755793151678"><richcontent TYPE="NODE">

<html>
  <head>
    
  </head>
  <body>
    <ul data-start="1027" data-end="1204">
      <li data-start="1139" data-end="1204">
        <p data-start="1141" data-end="1204">
          Trained for <strong data-start="1153" data-end="1166">10 epochs</strong>&#xa0;with same environment for fairness.
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
<node TEXT="Evaluation Metrics" ID="ID_981182879" CREATED="1755793160732" MODIFIED="1755793168048" STYLE="bubble">
<node TEXT="Accuracy, Precision, Recall, F1-score, MCC (Matthews Correlation Coefficient), MSE (Mean Squared Error)." ID="ID_56970460" CREATED="1755793168053" MODIFIED="1755793176321"/>
<node TEXT="Also reported inference time and model parameters." ID="ID_1855401512" CREATED="1755793176675" MODIFIED="1755793192287"/>
</node>
<node TEXT="Results" ID="ID_86651064" CREATED="1755793192619" MODIFIED="1755793202300" STYLE="bubble">
<node TEXT="Xception performed best (Accuracy ≈ 99.43%)." ID="ID_1793802836" CREATED="1755793202306" MODIFIED="1755793208338"/>
<node TEXT="MobileNetV2 performed well (lighter, but reported latency inconsistency)." ID="ID_49110985" CREATED="1755793208577" MODIFIED="1755793215550"/>
<node TEXT="ResNet50V2 moderate results." ID="ID_414031319" CREATED="1755793215826" MODIFIED="1755793221071"/>
<node TEXT="VGG16 showed mismatched metrics (accuracy high but macro-F1 low)." ID="ID_44472704" CREATED="1755793221325" MODIFIED="1755793226266"/>
</node>
</node>
</node>
</map>
