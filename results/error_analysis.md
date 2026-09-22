# Error Analysis Report

## Misclassified Examples

### ngram

- **[bangla]** "প্রোডাক্টটি ভালো, আপনারা নিতে পারেন।"
  true=`neutral`, predicted=`positive` -- Predicted the majority class ('positive') -- possible class-imbalance bias.
- **[banglish]** "Smell ta beshi Valo lage ni Amar kache. Tobe ator ta Valo. Al Nuaim er Ator gulo ja bujhechi ta holo beshikhon stay kore na. Tobe ator Valo. thank you. 😊😊"
  true=`positive`, predicted=`neutral` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "product er man ভালোনা"
  true=`positive`, predicted=`negative` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.
- **[english]** "Good bundle"
  true=`neutral`, predicted=`positive` -- Predicted the majority class ('positive') -- possible class-imbalance bias.
- **[english]** "Normal & Local Packed.."
  true=`neutral`, predicted=`negative` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.

### bow

- **[bangla]** "এইখানে ডেট টা পুরা মুছে গেছে,,,
  দেখে মনে হচ্ছে পুরান প্রডাক্ট
  আমি কি করে বুঝব এটা ডেট কবে শেষ হবে"
  true=`neutral`, predicted=`negative` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "fana kom hoi...product ta arro valo howa dorker."
  true=`neutral`, predicted=`negative` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "order date: 25/03/23
  delivary date:28/03/23
  WRITE SPEED--34--38 MB
  READ SPEED--73-75 MB
  TOTAL STORAGE--126 GB

offer price delivary soho(pickup point) :1305 tk
আমি মনে করেছিলাম ১১৬-১১৮ পাব কখনো আশা করিনি ১২৮ এ ১২৬ জিবি স্পেস পাব।"
true=`positive`, predicted=`neutral` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.

- **[english]** "The product was worth of it’s price.
  That delivery man Mr. Shah Paran was very good. He was a pure gentle man 👌👌."
  true=`neutral`, predicted=`positive` -- Predicted the majority class ('positive') -- possible class-imbalance bias.
- **[banglish]** "baje cilo product ta"
  true=`positive`, predicted=`negative` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).

### tfidf

- **[bangla]** "খেব একটা ভাল না। মাঝে মাঝেই ইন্টারনেট ছেড়ে দেয়। দামের তুলনায় আরো ভাল আশা করেছিলাম।"
  true=`negative`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "Dam double keo nien na
  70 taka bitore pele nien
  product onujai eta 150 taka deserve kore na
  70 taka hole satisfied hotam✔️"
  true=`neutral`, predicted=`negative` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "আমি কি অর্ডার দিলাম আর আপনারা এটা কি ভেলিভারি দিলেন?
  You're joking with us? After long time, you delivered this and the item is not correct! what kind of joke is this?"
  true=`positive`, predicted=`negative` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.
- **[english]** "good product . good packing. like a shampoo"
  true=`neutral`, predicted=`positive` -- Predicted the majority class ('positive') -- possible class-imbalance bias.
- **[english]** "It's good for skin...."
  true=`positive`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.

### ann

- **[bangla]** "খোলা বতোল⁉️🔸
  এটা আশা করা যায় না😢
  দারাজ কর্তৃপক্ষ সচেতন হওয়া দরকার।
  অনেক টা পরিমাণে কম ছিল 🥀"
  true=`negative`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "power kom"
  true=`negative`, predicted=`neutral` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "Dam হিসাব খুব ভালো but speaker er sound onak kom...."
  true=`positive`, predicted=`neutral` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.
- **[english]** "Got the wrong product, ordered the taylor Swift stickers, and got the mini movie posters instead."
  true=`neutral`, predicted=`negative` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "sm to sm product e paisi... use krer pr amr mon e hoi akdom clean hoi nh blackhead type er jei gula asey oi gula thaika e jai e... 10/7 choley r ki"
  true=`positive`, predicted=`negative` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).

### rnn

- **[bangla]** "ঠিক আছে ভালো জিনিস"
  true=`neutral`, predicted=`positive` -- Predicted the majority class ('positive') -- possible class-imbalance bias.
- **[banglish]** "product hate paisi
  dekha jak koto toko service dei ar koto din Jai
  Akhon valo motoi coltese👍
  ar packeting ta akdom baje silo 👎"
  true=`neutral`, predicted=`positive` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "বেশি ভালো না 😴
  smell Don't last long"
  true=`positive`, predicted=`neutral` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.
- **[english]** "very very small"
  true=`neutral`, predicted=`negative` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[english]** "Good product. But delivery was poorly. they didn't deliver on right place"
  true=`positive`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.

### lstm

- **[bangla]** "এইবারের প্যাকেজিং বাজে ছিল। যাস্ট একটা পলিটে প্যাক করে দিয়ে দিছে। খুলে দেখি অনেকটা লিক করছে।
  আগের মত বক্সে পাঠাবেন আশা করি।"
  true=`neutral`, predicted=`negative` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "vai atar sata kono nat nay ke vaba set korbo... r plastic o onek normol"
  true=`neutral`, predicted=`positive` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "খারাপ না. Bachchar posondo hoese"
  true=`positive`, predicted=`neutral` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.
- **[english]** "Looks good. But make no mistake - this is just flexible plastic. Too much price for a plastic case."
  true=`neutral`, predicted=`negative` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "body rate 560tk daraz price 660tk.body price theka kivaba basi dam a sale kortasan apnara ."
  true=`negative`, predicted=`positive` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).

### attention

- **[bangla]** "প্রডাক্ট এর মান অনুযায়ী দাম বেশি। এই গাড়ি মার্কেট এ ১২০ টাকায় পাওয়া যাচ্ছে। আর অর্ডার করলাম লাল গাড়ি দিলেন অন্য কালারের গাড়ি।"
  true=`neutral`, predicted=`negative` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "10 taka er jinish"
  true=`negative`, predicted=`positive` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "The consistent problem with DARAZ is that you order one thing and the supplier supplies another.
  I ordered for XXL size Boxer briefs and in the same package one is of smaller size (XL).

I am being convinced that this is intentional.

কেউ অসৎ উদ্দেশ্যে নিয়ে ইচ্ছা করে ক্রেতা সাধারণ কে ক্ষতিগ্ৰস্ত করে দারাজের বদনাম করে বিশেষ কোনো স্বাথ হাসিলের চেষ্টা করছে।"
true=`positive`, predicted=`negative` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.

- **[english]** "Quality is NOT good as I expected as per price. Within a week its removed autocratically from my wooden doors."
  true=`negative`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[bangla]** "smell একদমই ভালনা। আরাবিয়ান উদ এর মতই কিন্তু ভালনা।"
  true=`negative`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.

### transformer

- **[bangla]** "মোটামুটি ভালো অতটাও ভালো না।
  দাম অনুযায়ী ঠিকই আছে। আমি অর্ডার করছিলাম আলাদা আলাদাভাবে একই সেলার আলাদা আলাদা পণ্য দিছে একটি অরজিনাল আরেকটা ডুপ্লিকেট"
  true=`negative`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "Price hisabe thik e chilo"
  true=`positive`, predicted=`neutral` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "Dam হিসাব খুব ভালো but speaker er sound onak kom...."
  true=`positive`, predicted=`neutral` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.
- **[english]** "product look orginal but is it orginal??"
  true=`negative`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[bangla]** "ভালো করে মুখ না আটাকানোতে ২ বোতলের অনেকাংশ পড়ে গেছে প্যাকেটের মধ্যে আর বাইরে। সেলারকে আরও সতর্ক হতে হবে আর খালি পলি ব্যাগে দিছে। ভালো করে প্যাকিংওও করতে হবে।"
  true=`negative`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.

### bert

- **[bangla]** "স্মেল ভালো তবে বেশিক্ষণ থাকে না। রিকমেন্ড করি না"
  true=`negative`, predicted=`neutral` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.
- **[banglish]** "package ta kharap babe deua silo,,,Poli ta sire gesilo.Khub e kharap eta"
  true=`neutral`, predicted=`negative` -- Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on).
- **[code_switched]** "Packaging and Delivery Good, কিন্তু Product এর perfume's liquid কম ছিল।"
  true=`neutral`, predicted=`positive` -- Code-switched input -- likely an unseen intra-sentence language-mixing pattern.
- **[english]** "packaging was good, got the product on time."
  true=`neutral`, predicted=`positive` -- Predicted the majority class ('positive') -- possible class-imbalance bias.
- **[bangla]** "বাজে কাজ করে নি"
  true=`neutral`, predicted=`negative` -- Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing.

## Text-Completion Bonus (Phase 9, qualitative only)

- **[lstm]** prompt="this product is"
  -> "this product is team akane dropping price very akto let to অবশ্যই vangse nei mode ruined harsh উপযোগী"
- **[lstm]** prompt="ei jinis ta"
  -> "ei jinis ta bt rusted amr ধরা বল ase কষ্টকর food aage cai ধুতে shikar guli made legeche"
- **[lstm]** prompt="পণ্যটি"
  -> "পণ্যটি caps thnk asar করছে দিব ২০৫ ১৫ বলার hoyejay তৈরি বাজারে whisper কেনার shipped প্রোডাক্ট"
- **[lstm]** prompt="product ta khub"
  -> "product ta khub দারাজকে truth satisfy message baj ki therefore do cholbe চালিয়েছি totota <unk> ar জিবির ta"
- **[transformer]** prompt="this product is"
  -> "this product is high to was we কিনলে and rate product the also it and not the very"
- **[transformer]** prompt="ei jinis ta"
  -> "ei jinis ta ভালো thik দিয়ে vloe খুব product product product ভালো সুন্দর na watt valo e তবে"
- **[transformer]** prompt="পণ্যটি"
  -> "পণ্যটি হয়ে ok after valo a product অনেক product ভালো good এবং dam product জন্য তুলনায়"
- **[transformer]** prompt="product ta khub"
  -> "product ta khub fit sundhor ভালো আছে febric ভালো একটা <unk> smell ভালো mota valo স্ট্রিং খারাপ করা"
