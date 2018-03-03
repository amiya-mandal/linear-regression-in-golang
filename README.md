# linear-regression-in-golang
## linear-regression-in-golang
this basic linear regression with gradient descent 
data set is form https://www.kaggle.com/andonians/random-linear-regression

```
pred_b= 0.014294470585446532  pre_m= 0.998762992026086
error rate::>> 6.463059826411041
predicted value:: 0.014294470585446532 , actual value:: 0 , error rate>> -Inf
predicted value:: 76.91904485659406 , actual value:: 79.77515201 , error rate>> 3.5801964414281833
predicted value:: 20.988317303133254 , actual value:: 23.17727887 , error rate>> 9.444428654219946
predicted value:: 21.987080295159338 , actual value:: 25.60926156 , error rate>> 14.14402854355736
predicted value:: 19.989554311107167 , actual value:: 17.85738813 , error rate>> -11.939966615415479
predicted value:: 35.969762183524544 , actual value:: 41.84986439 , error rate>> 14.050468961329544
predicted value:: 14.995739350976736 , actual value:: 9.805234876 , error rate>> -52.93605447108044
predicted value:: 61.937599976202776 , actual value:: 58.87465933 , error rate>> -5.2024770606902395
predicted value:: 94.89677871306361 , actual value:: 97.61793701 , error rate>> 2.7875597254812283
predicted value:: 19.989554311107167 , actual value:: 18.39512747 , error rate>> -8.667658561798312
predicted value:: 5.008109430715876 , actual value:: 8.746747654 , error rate>> 42.74318147928272
predicted value:: 4.00934643868979 , actual value:: 2.811415826 , error rate>> -42.60951374077489
predicted value:: 18.99079131908108 , actual value:: 17.09537241 , error rate>> -11.087321548914302
predicted value:: 95.8955417050897 , actual value:: 95.14907176 , error rate>> -0.7845267760179147
predicted value:: 61.937599976202776 , actual value:: 61.38800663 , error rate>> -0.8952780459468332
predicted value:: 35.969762183524544 , actual value:: 40.24701716 , error rate>> 10.627508019964417
predicted value:: 14.995739350976736 , actual value:: 14.82248589 , error rate>> -1.168855630981725
predicted value:: 64.93388895228102 , actual value:: 66.95806869 , error rate>> 3.0230557381970704
predicted value:: 13.996976358950649 , actual value:: 16.63507984 , error rate>> 15.858676402056577
predicted value:: 86.90667477685493 , actual value:: 90.65513736 , error rate>> 4.134859526228035
predicted value:: 68.92894092038537 , actual value:: 77.22982636 , error rate>> 10.748289658092443
predicted value:: 88.90420076090709 , actual value:: 92.11906278 , error rate>> 3.4898987485040776
predicted value:: 50.95120706391583 , actual value:: 46.91387709 , error rate>> -8.605833123044983
predicted value:: 88.90420076090709 , actual value:: 89.82634442 , error rate>> 1.0265848677769338
predicted value:: 26.98089525528977 , actual value:: 21.71380347 , error rate>> -24.25688245989164
predicted value:: 96.89430469711579 , actual value:: 97.41206981 , error rate>> 0.5315204921670476
predicted value:: 57.942548008098434 , actual value:: 57.01631363 , error rate>> -1.6245076525099704
predicted value:: 78.91657084064623 , actual value:: 78.31056542 , error rate>> -0.773848863682776
predicted value:: 20.988317303133254 , actual value:: 19.1315097 , error rate>> -9.705494402949576
predicted value:: 92.89925272901144 , actual value:: 93.03483388 , error rate>> 0.14573159894436408
predicted value:: 26.98089525528977 , actual value:: 26.59112396 , error rate>> -1.4657947361536392
predicted value:: 98.89183068116795 , actual value:: 97.55155344 , error rate>> -1.3739168612955912
predicted value:: 30.97594722339411 , actual value:: 31.43524822 , error rate>> 1.4611018605339536
predicted value:: 32.97347320744628 , actual value:: 35.12724777 , error rate>> 6.1313501605813805
predicted value:: 79.91533383267232 , actual value:: 78.61042432 , error rate>> -1.6599700662604313
predicted value:: 27.979658247315854 , actual value:: 33.07112825 , error rate>> 15.395513464782221
predicted value:: 46.95615509581149 , actual value:: 51.69967172 , error rate>> 9.17513877047208
predicted value:: 52.948733047968005 , actual value:: 53.62235225 , error rate>> 1.2562283707574453
predicted value:: 68.92894092038537 , actual value:: 69.46306072 , error rate>> 0.7689263819911752
predicted value:: 27.979658247315854 , actual value:: 27.42497237 , error rate>> -2.0225576523191764
predicted value:: 32.97347320744628 , actual value:: 36.34644189 , error rate>> 9.280051931250314
predicted value:: 90.90172674495926 , actual value:: 95.06140858 , error rate>> 4.375783924493519
predicted value:: 70.92646690443755 , actual value:: 68.16724757 , error rate>> -4.047720030949089
predicted value:: 49.952444071889744 , actual value:: 50.96155532 , error rate>> 1.9801421714345322
predicted value:: 75.92028186456797 , actual value:: 78.04237454 , error rate>> 2.719154418276147
predicted value:: 4.00934643868979 , actual value:: 5.607664865 , error rate>> 28.50238851265963
predicted value:: 36.96852517555063 , actual value:: 36.11334779 , error rate>> -2.368036855855929
predicted value:: 69.92770391241146 , actual value:: 67.2352155 , error rate>> -4.004580623990807
predicted value:: 67.93017792835929 , actual value:: 65.01324035 , error rate>> -4.486682347558578
predicted value:: 39.964814151628886 , actual value:: 38.14753871 , error rate>> -4.763807844705079
predicted value:: 34.97099919149846 , actual value:: 34.31141446 , error rate>> -1.9223478305372528
predicted value:: 93.89801572103752 , actual value:: 95.28503937 , error rate>> 1.4556573184343775
predicted value:: 87.905437768881 , actual value:: 87.84749912 , error rate>> -0.06595366909860628
predicted value:: 51.94997005594192 , actual value:: 54.08170635 , error rate>> 3.941695700690626
predicted value:: 30.97594722339411 , actual value:: 31.93063515 , error rate>> 2.989880790410424
predicted value:: 58.94131100012452 , actual value:: 59.61247085 , error rate>> 1.1258715505423134
predicted value:: 0.014294470585446532 , actual value:: -1.040114209 , error rate>> 101.37431740300806
predicted value:: 38.9660511596028 , actual value:: 47.49374765 , error rate>> 17.955408685036048
predicted value:: 63.93512596025495 , actual value:: 62.60089773 , error rate>> -2.1313244356487133
predicted value:: 68.92894092038537 , actual value:: 70.9146434 , error rate>> 2.800130388323479
predicted value:: 56.94378501607235 , actual value:: 56.14834113 , error rate>> -1.4166827907358133
predicted value:: 12.998213366924563 , actual value:: 14.05572877 , error rate>> 7.523732282971738
predicted value:: 71.92522989646363 , actual value:: 68.11367147 , error rate>> -5.595878689555588
predicted value:: 75.92028186456797 , actual value:: 75.59701346 , error rate>> -0.4276206026829582
predicted value:: 60.938836984176696 , actual value:: 59.225745 , error rate>> -2.8924785735944605
predicted value:: 81.91285981672449 , actual value:: 85.45504157 , error rate>> 4.1450822423086136
predicted value:: 17.992028327054996 , actual value:: 17.76197116 , error rate>> -1.2952231764292215
predicted value:: 40.96357714365497 , actual value:: 38.68888682 , error rate>> -5.87944112798584
predicted value:: 49.952444071889744 , actual value:: 50.96343637 , error rate>> 1.9837600643142284
predicted value:: 54.94625903202018 , actual value:: 51.83503872 , error rate>> -6.002156820652183
predicted value:: 12.998213366924563 , actual value:: 17.0761107 , error rate>> 23.88071502180785
predicted value:: 45.9573921037854 , actual value:: 46.56141773 , error rate>> 1.2972663970784126
predicted value:: 12.998213366924563 , actual value:: 10.34754461 , error rate>> -25.616403280474124
predicted value:: 78.91657084064623 , actual value:: 77.91032969 , error rate>> -1.2915375337904471
predicted value:: 52.948733047968005 , actual value:: 50.17008622 , error rate>> -5.538453363989462
predicted value:: 14.995739350976736 , actual value:: 13.25690647 , error rate>> -13.116430178576461
predicted value:: 27.979658247315854 , actual value:: 31.32274932 , error rate>> 10.673044816501907
predicted value:: 80.9140968246984 , actual value:: 73.9308764 , error rate>> -9.445607525218518
predicted value:: 68.92894092038537 , actual value:: 74.45114379 , error rate>> 7.417216967399167
predicted value:: 51.94997005594192 , actual value:: 52.01932286 , error rate>> 0.13332123573529425
predicted value:: 83.91038580077667 , actual value:: 83.68820499 , error rate>> -0.26548640970757115
predicted value:: 67.93017792835929 , actual value:: 70.3698748 , error rate>> 3.466962075141734
predicted value:: 26.98089525528977 , actual value:: 23.44479161 , error rate>> -15.082683199374237
predicted value:: 55.94502202404626 , actual value:: 49.83051801 , error rate>> -12.27060094542706
predicted value:: 47.95491808783758 , actual value:: 49.88226593 , error rate>> 3.863793687454138
predicted value:: 39.964814151628886 , actual value:: 41.04525583 , error rate>> 2.6323180511922177
predicted value:: 38.9660511596028 , actual value:: 33.37834391 , error rate>> -16.740516739444192
predicted value:: 81.91285981672449 , actual value:: 81.29750133 , error rate>> -0.7569217708508001
predicted value:: 99.89059367319403 , actual value:: 105.5918375 , error rate>> 5.399322487219682
predicted value:: 58.94131100012452 , actual value:: 56.82457013 , error rate>> -3.7250451086246055
predicted value:: 42.96110312770715 , actual value:: 48.67252645 , error rate>> 11.734388450453762
predicted value:: 66.9314149363332 , actual value:: 67.02150613 , error rate>> 0.13442132066095372
predicted value:: 37.96728816757671 , actual value:: 38.43076389 , error rate>> 1.2060018472437652
predicted value:: 62.93636296822886 , actual value:: 58.61466887 , error rate>> -7.373058965519087
predicted value:: 90.90172674495926 , actual value:: 89.12377509 , error rate>> -1.9949240852554053
predicted value:: 59.94007399215061 , actual value:: 60.9105427 , error rate>> 1.5932688576248588
predicted value:: 13.996976358950649 , actual value:: 13.83959878 , error rate>> -1.137154201161383
predicted value:: 20.988317303133254 , actual value:: 16.89085185 , error rate>> -24.258489089366165
predicted value:: 86.90667477685493 , actual value:: 84.06676818 , error rate>> -3.378156028044578
predicted value:: 72.92399288848972 , actual value:: 70.34969772 , error rate>> -3.6592839087038036
predicted value:: 31.9747102154202 , actual value:: 33.38474138 , error rate>> 4.223579714247894
predicted value:: 2.0118204546376184 , actual value:: -1.63296825 , error rate>> 223.20021865934126
predicted value:: 81.91285981672449 , actual value:: 88.54475895 , error rate>> 7.489883322196915
predicted value:: 18.99079131908108 , actual value:: 17.44047622 , error rate>> -8.889178710058633
predicted value:: 73.92275588051581 , actual value:: 75.69298554 , error rate>> 2.3386971023209386
predicted value:: 41.96234013568106 , actual value:: 41.97607107 , error rate>> 0.0327113376000459
predicted value:: 11.999450374898478 , actual value:: 12.59244741 , error rate>> 4.70914839501817
predicted value:: 1.0130574626115325 , actual value:: 0.275307261 , error rate>> -267.9733905062288
predicted value:: 89.90296375293318 , actual value:: 98.13258005 , error rate>> 8.386222285069561
predicted value:: 88.90420076090709 , actual value:: 87.45721555 , error rate>> -1.6545063798421913
predicted value:: 0.014294470585446532 , actual value:: -2.344738542 , error rate>> 100.60964027883698
predicted value:: 40.96357714365497 , actual value:: 39.3294153 , error rate>> -4.155062645070577
predicted value:: 15.994502343002821 , actual value:: 16.68715211 , error rate>> 4.150796747289781
predicted value:: 93.89801572103752 , actual value:: 96.58888601 , error rate>> 2.7859005317484273
predicted value:: 96.89430469711579 , actual value:: 97.70342201 , error rate>> 0.8281361043847553
predicted value:: 65.93265194430711 , actual value:: 67.01715955 , error rate>> 1.6182536129179943
predicted value:: 23.984606279211512 , actual value:: 25.63476257 , error rate>> 6.437181878640226
predicted value:: 16.99326533502891 , actual value:: 13.41310757 , error rate>> -26.69148626703297
predicted value:: 89.90296375293318 , actual value:: 95.15647284 , error rate>> 5.520916160795806
predicted value:: 12.998213366924563 , actual value:: 9.744164258 , error rate>> -33.39485073081538
predicted value:: 0.014294470585446532 , actual value:: -3.467883789 , error rate>> 100.41219577861254
predicted value:: 63.93512596025495 , actual value:: 62.82816355 , error rate>> -1.7618888531956003
predicted value:: 95.8955417050897 , actual value:: 97.27405461 , error rate>> 1.4171434617762717
predicted value:: 97.89306768914187 , actual value:: 95.58017185 , error rate>> -2.419849006728767
predicted value:: 11.999450374898478 , actual value:: 7.468501839 , error rate>> -60.66743549875262
predicted value:: 40.96357714365497 , actual value:: 45.44599591 , error rate>> 9.863176450620395
predicted value:: 46.95615509581149 , actual value:: 46.69013968 , error rate>> -0.5697464553215656
predicted value:: 77.91780784862014 , actual value:: 74.4993599 , error rate>> -4.588560161065412
predicted value:: 19.989554311107167 , actual value:: 21.63500655 , error rate>> 7.605508392567753
predicted value:: 88.90420076090709 , actual value:: 91.59548851 , error rate>> 2.938231776338072
predicted value:: 28.97842123934194 , actual value:: 26.49487961 , error rate>> -9.373666406110306
predicted value:: 63.93512596025495 , actual value:: 67.38654703 , error rate>> 5.121825085070029
predicted value:: 74.9215188725419 , actual value:: 74.25362837 , error rate>> -0.8994718739047313
predicted value:: 11.999450374898478 , actual value:: 12.07991648 , error rate>> 0.6661147470244897
predicted value:: 24.983369271237596 , actual value:: 21.32273728 , error rate>> -17.16773950346021
predicted value:: 27.979658247315854 , actual value:: 29.31770045 , error rate>> 4.563939811603289
predicted value:: 29.977184231368028 , actual value:: 26.48713683 , error rate>> -13.1763860464341
predicted value:: 64.93388895228102 , actual value:: 68.94699774 , error rate>> 5.820570756180655
predicted value:: 58.94131100012452 , actual value:: 59.10598995 , error rate>> 0.2786163466930984
predicted value:: 63.93512596025495 , actual value:: 64.37521087 , error rate>> 0.6836248049482361
predicted value:: 52.948733047968005 , actual value:: 60.20758349 , error rate>> 12.056372339271231
predicted value:: 70.92646690443755 , actual value:: 70.34329706 , error rate>> -0.8290339930187359
predicted value:: 96.89430469711579 , actual value:: 97.1082562 , error rate>> 0.220322670034944
predicted value:: 72.92399288848972 , actual value:: 75.7584178 , error rate>> 3.741399297690035
predicted value:: 9.00316139882022 , actual value:: 10.80462727 , error rate>> 16.67309594456542
predicted value:: 11.999450374898478 , actual value:: 12.11219941 , error rate>> 0.9308716879977669
predicted value:: 62.93636296822886 , actual value:: 63.28312382 , error rate>> 0.5479515403782053
predicted value:: 98.89183068116795 , actual value:: 98.03017721 , error rate>> -0.87896757477252
predicted value:: 59.94007399215061 , actual value:: 63.19354354 , error rate>> 5.1484208126262505
predicted value:: 34.97099919149846 , actual value:: 34.8534823 , error rate>> -0.33717403181389966
predicted value:: 2.0118204546376184 , actual value:: -2.819913974 , error rate>> 171.3433272499404
predicted value:: 59.94007399215061 , actual value:: 59.8313966 , error rate>> -0.18163940393564337
predicted value:: 31.9747102154202 , actual value:: 29.38505024 , error rate>> -8.812848554858205
predicted value:: 93.89801572103752 , actual value:: 97.00148372 , error rate>> 3.1994026069959913
predicted value:: 83.91038580077667 , actual value:: 85.18657275 , error rate>> 1.498108103220211
predicted value:: 62.93636296822886 , actual value:: 61.74063192 , error rate>> -1.936700372257649
predicted value:: 21.987080295159338 , actual value:: 18.84798163 , error rate>> -16.65482663757954
predicted value:: 80.9140968246984 , actual value:: 78.79008525 , error rate>> -2.695785349081597
predicted value:: 92.89925272901144 , actual value:: 95.12400481 , error rate>> 2.338791439061328
predicted value:: 32.97347320744628 , actual value:: 30.48881287 , error rate>> -8.149416469708164
predicted value:: 7.005635414768048 , actual value:: 10.41468095 , error rate>> 32.73307700541658
predicted value:: 41.96234013568106 , actual value:: 38.98317436 , error rate>> -7.642183646126915
predicted value:: 45.9573921037854 , actual value:: 46.11021062 , error rate>> 0.3314201218337322
predicted value:: 53.94749603999409 , actual value:: 52.45103628 , error rate>> -2.8530604276444165
predicted value:: 15.994502343002821 , actual value:: 21.16523945 , error rate>> 24.43032652294033
predicted value:: 48.953681079863664 , actual value:: 52.28620611 , error rate>> 6.373621798310158
predicted value:: 42.96110312770715 , actual value:: 44.18863945 , error rate>> 2.77794550267116
predicted value:: 94.89677871306361 , actual value:: 97.13832018 , error rate>> 2.3075769302812157
predicted value:: 65.93265194430711 , actual value:: 67.22008001 , error rate>> 1.9152432807300568
predicted value:: 20.988317303133254 , actual value:: 18.98322306 , error rate>> -10.562454209149738
predicted value:: 34.97099919149846 , actual value:: 24.3884599 , error rate>> -43.39158493357121
predicted value:: 79.91533383267232 , actual value:: 79.44769523 , error rate>> -0.5886119179650413
predicted value:: 36.96852517555063 , actual value:: 40.03504862 , error rate>> 7.659597153373873
predicted value:: 53.94749603999409 , actual value:: 53.32005764 , error rate>> -1.1767399132055605
predicted value:: 55.94502202404626 , actual value:: 54.55446979 , error rate>> -2.5489244774974487
predicted value:: 1.0130574626115325 , actual value:: -2.761182595 , error rate>> 136.6892600455325
predicted value:: 31.9747102154202 , actual value:: 37.80182795 , error rate>> 15.414909941094011
predicted value:: 57.942548008098434 , actual value:: 57.48741435 , error rate>> -0.7917100868851873
predicted value:: 31.9747102154202 , actual value:: 36.06292994 , error rate>> 11.336349352039914
predicted value:: 45.9573921037854 , actual value:: 49.83538167 , error rate>> 7.781599009101348
predicted value:: 71.92522989646363 , actual value:: 74.68953276 , error rate>> 3.7010579145258693
predicted value:: 16.99326533502891 , actual value:: 14.86159401 , error rate>> -14.343490500376747
predicted value:: 96.89430469711579 , actual value:: 101.0697879 , error rate>> 4.131287192385816
predicted value:: 92.89925272901144 , actual value:: 99.43577876 , error rate>> 6.573615767384139
predicted value:: 90.90172674495926 , actual value:: 91.69240746 , error rate>> 0.8623186335091741
predicted value:: 36.96852517555063 , actual value:: 34.12473248 , error rate>> -8.333523778442329
predicted value:: 4.00934643868979 , actual value:: 6.079390073 , error rate>> 34.0501861116588
predicted value:: 53.94749603999409 , actual value:: 59.07247174 , error rate>> 8.675742776707967
predicted value:: 50.95120706391583 , actual value:: 56.43046022 , error rate>> 9.709743877194573
predicted value:: 26.98089525528977 , actual value:: 30.49412933 , error rate>> 11.52101782179406
predicted value:: 45.9573921037854 , actual value:: 48.35172635 , error rate>> 4.951910566507824
predicted value:: 91.90048973698535 , actual value:: 89.73153611 , error rate>> -2.4171586947162944
predicted value:: 72.92399288848972 , actual value:: 72.86282528 , error rate>> -0.08394899354323455
predicted value:: 76.91904485659406 , actual value:: 80.97144285 , error rate>> 5.004724938535471
predicted value:: 90.90172674495926 , actual value:: 91.36566374 , error rate>> 0.5077804681209006
predicted value:: 60.938836984176696 , actual value:: 60.07137496 , error rate>> -1.4440522208028646
predicted value:: 98.89183068116795 , actual value:: 99.87382707 , error rate>> 0.983236967723078
predicted value:: 4.00934643868979 , actual value:: 8.655714172 , error rate>> 53.67977316465169
predicted value:: 71.92522989646363 , actual value:: 69.39858505 , error rate>> -3.6407728553013774
predicted value:: 18.99079131908108 , actual value:: 19.38780134 , error rate>> 2.0477310137267986
predicted value:: 56.94378501607235 , actual value:: 53.11628433 , error rate>> -7.205889369619518
predicted value:: 77.91780784862014 , actual value:: 78.39683006 , error rate>> 0.6110224240103113
predicted value:: 25.982132263263683 , actual value:: 25.75612514 , error rate>> -0.8774888382285639
predicted value:: 73.92275588051581 , actual value:: 75.07484683 , error rate>> 1.5345898102103264
predicted value:: 89.90296375293318 , actual value:: 92.88772282 , error rate>> 3.21329770657717
predicted value:: 65.93265194430711 , actual value:: 69.45498498 , error rate>> 5.071389817026342
predicted value:: 12.998213366924563 , actual value:: 13.12109842 , error rate>> 0.9365454715904474
predicted value:: 39.964814151628886 , actual value:: 48.09843134 , error rate>> 16.910358533058787
predicted value:: 76.91904485659406 , actual value:: 79.3142548 , error rate>> 3.0198984400039333
predicted value:: 66.9314149363332 , actual value:: 68.48820749 , error rate>> 2.273081178090555
predicted value:: 74.9215188725419 , actual value:: 73.2300846 , error rate>> -2.3097532684564146
predicted value:: 22.985843287185425 , actual value:: 24.68362712 , error rate>> 6.878178091739768
predicted value:: 44.958629111759315 , actual value:: 41.90368917 , error rate>> -7.2903842174030595
predicted value:: 58.94131100012452 , actual value:: 62.22635684 , error rate>> 5.279187159103945
predicted value:: 43.95986611973323 , actual value:: 45.96396877 , error rate>> 4.3601601513027335
predicted value:: 22.985843287185425 , actual value:: 23.52647153 , error rate>> 2.297957184634281
predicted value:: 54.94625903202018 , actual value:: 51.80035866 , error rate>> -6.073124691411508
predicted value:: 54.94625903202018 , actual value:: 51.10774273 , error rate>> -7.510635565141074
predicted value:: 94.89677871306361 , actual value:: 95.79747345 , error rate>> 0.9402071938843886
predicted value:: 11.999450374898478 , actual value:: 9.241138977 , error rate>> -29.848175693099716
predicted value:: 4.00934643868979 , actual value:: 7.646529763 , error rate>> 47.5664574263452
predicted value:: 7.005635414768048 , actual value:: 9.281699753 , error rate>> 24.522063833149634
predicted value:: 99.89059367319403 , actual value:: 103.5266162 , error rate>> 3.5121620509470235
predicted value:: 47.95491808783758 , actual value:: 47.41006725 , error rate>> -1.1492302572879796
predicted value:: 41.96234013568106 , actual value:: 42.03835773 , error rate>> 0.18082912469411766
predicted value:: 95.8955417050897 , actual value:: 96.11982476 , error rate>> 0.2333369369641596
predicted value:: 38.9660511596028 , actual value:: 38.05766408 , error rate>> -2.3868702968561086
predicted value:: 99.89059367319403 , actual value:: 105.4503788 , error rate>> 5.272418354561627
predicted value:: 86.90667477685493 , actual value:: 88.80306911 , error rate>> 2.1355053965488646
predicted value:: 13.996976358950649 , actual value:: 15.49301141 , error rate>> 9.656192792085154
predicted value:: 13.996976358950649 , actual value:: 12.42624606 , error rate>> -12.640424882674893
predicted value:: 36.96852517555063 , actual value:: 40.00709598 , error rate>> 7.595079647791448
predicted value:: 5.008109430715876 , actual value:: 5.634030902 , error rate>> 11.109656339689769
predicted value:: 87.905437768881 , actual value:: 87.36938931 , error rate>> -0.6135426413237433
predicted value:: 90.90172674495926 , actual value:: 89.73951993 , error rate>> -1.295089182409071
predicted value:: 64.93388895228102 , actual value:: 66.61499643 , error rate>> 2.5236171550133064
predicted value:: 73.92275588051581 , actual value:: 72.9138853 , error rate>> -1.3836467174460174
predicted value:: 55.94502202404626 , actual value:: 57.19103506 , error rate>> 2.1786859332874915
predicted value:: 15.994502343002821 , actual value:: 11.21710477 , error rate>> -42.59029108634081
predicted value:: 5.008109430715876 , actual value:: 0.676076749 , error rate>> -640.7604888237144
predicted value:: 27.979658247315854 , actual value:: 28.15668543 , error rate>> 0.6287216694033502
predicted value:: 91.90048973698535 , actual value:: 95.3958003 , error rate>> 3.6640088473733945
predicted value:: 45.9573921037854 , actual value:: 52.05490703 , error rate>> 11.713621777675089
predicted value:: 53.94749603999409 , actual value:: 59.70864577 , error rate>> 9.648769714520197
predicted value:: 38.9660511596028 , actual value:: 36.79224762 , error rate>> -5.908319497233263
predicted value:: 43.95986611973323 , actual value:: 37.08457698 , error rate>> -18.539483795220647
predicted value:: 30.97594722339411 , actual value:: 24.18437976 , error rate>> -28.082454587597464
predicted value:: 67.93017792835929 , actual value:: 67.28725332 , error rate>> -0.9554924248456157
predicted value:: 85.90791178482884 , actual value:: 82.870594 , error rate>> -3.6651333581956025
predicted value:: 89.90296375293318 , actual value:: 89.899991 , error rate>> -0.0033067332934161426
predicted value:: 37.96728816757671 , actual value:: 36.94173178 , error rate>> -2.7761459416256793
predicted value:: 20.988317303133254 , actual value:: 19.87562242 , error rate>> -5.598289500677961
predicted value:: 94.89677871306361 , actual value:: 90.71481654 , error rate>> -4.610010065135948
predicted value:: 55.94502202404626 , actual value:: 61.09367762 , error rate>> 8.42747694446904
predicted value:: 59.94007399215061 , actual value:: 60.11134958 , error rate>> 0.28493053149879666
predicted value:: 64.93388895228102 , actual value:: 64.83296316 , error rate>> -0.15567049130847999
predicted value:: 77.91780784862014 , actual value:: 81.40381769 , error rate>> 4.282366528134086
predicted value:: 88.90420076090709 , actual value:: 92.40217686 , error rate>> 3.7855992336552293
predicted value:: 6.006872422741963 , actual value:: 2.576625376 , error rate>> -133.12944437685934
predicted value:: 66.9314149363332 , actual value:: 63.80768172 , error rate>> -4.895544129060704
predicted value:: 35.969762183524544 , actual value:: 38.67780759 , error rate>> 7.001548368981522
predicted value:: 15.994502343002821 , actual value:: 16.82839701 , error rate>> 4.955282826413295
predicted value:: 99.89059367319403 , actual value:: 99.78687252 , error rate>> -0.1039426836162658
predicted value:: 44.958629111759315 , actual value:: 44.68913433 , error rate>> -0.6030431911463549
predicted value:: 72.92399288848972 , actual value:: 71.00377824 , error rate>> -2.7043837610995816
predicted value:: 56.94378501607235 , actual value:: 51.57326718 , error rate>> -10.413375242115784
predicted value:: 19.989554311107167 , actual value:: 19.87846479 , error rate>> -0.5588435640314247
predicted value:: 75.92028186456797 , actual value:: 79.50341495 , error rate>> 4.506892046946014
predicted value:: 33.97223619947237 , actual value:: 34.58876491 , error rate>> 1.782453672838103
predicted value:: 54.94625903202018 , actual value:: 55.7383467 , error rate>> 1.4210821003412022
predicted value:: 71.92522989646363 , actual value:: 68.19721905 , error rate>> -5.466514468471765
predicted value:: 54.94625903202018 , actual value:: 55.81628509 , error rate>> 1.5587315719363313
predicted value:: 8.004398406794135 , actual value:: 9.391416798 , error rate>> 14.769000471805759
predicted value:: 55.94502202404626 , actual value:: 56.01448111 , error rate>> 0.12400201622386949
predicted value:: 71.92522989646363 , actual value:: 77.9969477 , error rate>> 7.784558220008875
predicted value:: 57.942548008098434 , actual value:: 55.37049953 , error rate>> -4.645160328930913
predicted value:: 6.006872422741963 , actual value:: 11.89457829 , error rate>> 49.49907196128125
predicted value:: 95.8955417050897 , actual value:: 94.79081712 , error rate>> -1.1654341830297523
predicted value:: 22.985843287185425 , actual value:: 25.69041546 , error rate>> 10.527553269917325
predicted value:: 57.942548008098434 , actual value:: 53.52042319 , error rate>> -8.262499723516166
predicted value:: 22.985843287185425 , actual value:: 18.31396758 , error rate>> -25.509904867841996
predicted value:: 18.99079131908108 , actual value:: 21.42637785 , error rate>> 11.367234107275495
predicted value:: 24.983369271237596 , actual value:: 30.41303282 , error rate>> 17.853081542041373
predicted value:: 63.93512596025495 , actual value:: 67.68142149 , error rate>> 5.535190377611341
predicted value:: 20.988317303133254 , actual value:: 17.0854783 , error rate>> -22.843018700467145
predicted value:: 58.94131100012452 , actual value:: 60.91792707 , error rate>> 3.2447198467606624
predicted value:: 18.99079131908108 , actual value:: 14.99514319 , error rate>> -26.646281922440775
predicted value:: 15.994502343002821 , actual value:: 16.74923937 , error rate>> 4.506097323732857
predicted value:: 41.96234013568106 , actual value:: 41.46923883 , error rate>> -1.1890773006528763
predicted value:: 42.96110312770715 , actual value:: 42.84526108 , error rate>> -0.2703730699431353
predicted value:: 60.938836984176696 , actual value:: 59.12912974 , error rate>> -3.060601859243079
predicted value:: 91.90048973698535 , actual value:: 91.30863673 , error rate>> -0.6481895121657086
predicted value:: 11.00068738287239 , actual value:: 8.673336357 , error rate>> -26.833399859951847
predicted value:: 40.96357714365497 , actual value:: 39.31485292 , error rate>> -4.19364210012406
predicted value:: 1.0130574626115325 , actual value:: 5.313686205 , error rate>> 80.93494001097994
predicted value:: 8.004398406794135 , actual value:: 5.405220518 , error rate>> -48.08643569931284
predicted value:: 70.92646690443755 , actual value:: 68.5458879 , error rate>> -3.4729712859078012
predicted value:: 45.9573921037854 , actual value:: 47.33487629 , error rate>> 2.910082996257039
predicted value:: 54.94625903202018 , actual value:: 54.09063686 , error rate>> -1.5818304639946208
predicted value:: 61.937599976202776 , actual value:: 63.29717058 , error rate>> 2.1479168679094274
predicted value:: 46.95615509581149 , actual value:: 52.45946688 , error rate>> 10.490598001648069
```
