from __future__ import annotations

from .config import HUMAN_REVIEW_CONFIDENCE_THRESHOLD


FALLBACK_REPLY = (
    "Mesajınızı aldık. Talebinizi daha net inceleyebilmemiz için lütfen problemi "
    "kısa ve somut detaylarla tekrar iletin."
)

REPLY_TEMPLATES: dict[str, str] = {
    "arkadas_adina_ban_itirazi": (
        "Sistem logları ve kayıtları açıktır. Başkası adına itirazda bulunamazsınız. "
        "İlgili kullanıcının kendi hesabıyla destek talebi oluşturması gerekir."
    ),
    "ban_ceza_itirazi": (
        "Ceza itirazınız kayıt altına alınmıştır. Moderasyon ekibi sistem loglarını "
        "inceleyerek gerekli kontrolü yapacaktır."
    ),
    "bilgi_sorusu": (
        "Ana ekranda yer alan ? butonuna basarak oyun hakkında detaylı bilgi alabilirsiniz. "
        "Bilet sistemi için profilinizdeki Koleksiyon alanındaki i butonunu kullanabilirsiniz."
    ),
    "davet_odulu": (
        "Davet edilen kişinin daha önce hesap açmamış olması gerekir. Ödül, davet ettiğiniz "
        "kullanıcının ilk oyununu tamamlamasının ardından hesabınıza otomatik yüklenir."
    ),
    "giris_hesap_erisim": (
        "Giriş sorununuz kayıt altına alınmıştır. Google, Apple veya Facebook hesabınızla "
        "ilgili erişim kontrolü için destek ekibi inceleme yapacaktır."
    ),
    "hediye_bonus_talebi": (
        "Sistemin otomatik verdiği hediyeler dışında manuel Manc Altın, çip, üyelik paketi "
        "veya bonus tanımlaması yapılmamaktadır."
    ),
    "hesap_islemleri": (
        "Hesap işlemleri talebiniz kayıt altına alınmıştır. Güvenlik nedeniyle gerekli "
        "kontroller destek ekibi tarafından yapılacaktır."
    ),
    "hesap_silme_kapatma": (
        "Profilinize girip profil düzenleme ekranındaki Hesabı Sil seçeneğiyle hesabınızı "
        "silebilirsiniz. Hesaba tekrar giriş yapmazsanız hesap 7 gün içinde kalıcı olarak silinir."
    ),
    "hile_itirazi": (
        "Hile itirazınız kayıt altına alınmıştır. Sistem logları ve oyun kayıtları moderasyon "
        "ekibi tarafından incelenecektir."
    ),
    "iade_tazmin_talebi": (
        "Çip iadesi talebiniz kayıt altına alınmıştır. Sistem tarafından silinen Manc Altın/çip "
        "görünmüyorsa iade işlemi yapılamamaktadır. Çiplerinizin silindiğini düşünüyorsanız "
        "işlem zamanı, yaklaşık tutar ve varsa ekran görüntüsüyle birlikte destek@mancgames.com "
        "adresine iletebilirsiniz."
    ),
    "kullanici_adi_degistirme": (
        "Kullanıcı adı değişikliği talebiniz alınmıştır. Değiştirmek istediğiniz kullanıcı adını "
        "iletebilirsiniz; uygunluk kontrolünden sonra değerlendirilecektir."
    ),
    "odeme_cip_yukleme": (
        "Satın alımınıza ait Google Play GPA kodlu veya Apple Store sipariş numarasını/dekont "
        "görüntüsünü destek@mancgames.com adresine iletebilirsiniz. Sistemimize ulaşan satın "
        "almalar otomatik yüklenir."
    ),
    "olumlu_geri_bildirim": "Geri bildiriminiz için teşekkür ederiz. İyi oyunlar dileriz.",
    "oyun_adalet_puan": (
        "Oyuna müdahalemiz yoktur. Kurallar ve oyun mekanikleri tüm kullanıcılar için eşittir; "
        "satın alma işlemi kazanma/kaybetme durumunu değiştirmez."
    ),
    "oyun_nasil_oynanir_yardim": (
        "Ana ekranda yer alan ? butonundan oyun hakkında detaylı bilgi alabilirsiniz. Elinizi "
        "açmak için taşlarınızı kurala uygun dizip sıra size geldiğinde fazla taşı ortadaki "
        "kapalı taşların üzerine bırakmanız gerekir."
    ),
    "oyuncu_sikayet_raporlama": (
        "Oyuncu şikayetlerinizi, şikayet etmek istediğiniz kullanıcının profilindeki Şikayet Et "
        "düğmesini kullanarak iletebilirsiniz."
    ),
    "ozel_direkt_mesaj": (
        "Diğer kullanıcılara mesaj atabilmek için Altın, Platin veya Elmas Manc kullanıcı "
        "hesabına sahip olmanız gerekir."
    ),
    "ozellik_degisim_talebi": (
        "Bildiriminiz kayıt altına alınmıştır. Özellik değişikliği öneriniz ilgili ekip "
        "tarafından değerlendirilmek üzere iletilecektir."
    ),
    "profil_fotografi_degistirme": (
        "Kullanıcı adı ve profil resmi, Google, Apple veya Facebook hesabınızla giriş yaptığınızda "
        "hesabınıza eklenebilir/güncellenebilir."
    ),
    "reklam_odulu": (
        "Reklam yayınları Google AdMob tarafından sağlanmaktadır. AdMob size reklam sundukça "
        "reklam izleyebilirsiniz. Reklam ödülü yüklenmediyse kısa süre sonra tekrar kontrol ediniz."
    ),
    "ses_sorunu": (
        "Telefon ayarlarınızdan ses ve mikrofon izinlerini kontrol edin. Ardından oyun içindeki "
        "ayarlar menüsünden ses ve mikrofonun aktif olduğundan emin olun."
    ),
    "sosyal_hesap_giris_gecis": (
        "Google, Apple veya Facebook ile girişlerde cihazınızda aktif olan doğru hesabı "
        "seçtiğinizden emin olun. Farklı hesaba geçmek için önce mevcut oturumdan çıkış yapmanız gerekir."
    ),
    "teknik_uygulama_sorunu": (
        "Hata bildiriminiz sistemimiz tarafından kayıt altına alınmıştır. Teknik ekibimiz "
        "tarafından inceleme başlatılmıştır. İyi oyunlar."
    ),
    "toplu_sikayet": (
        "Kullanıcılarla ilgili şikayetlerinizi, şikayet etmek istediğiniz kullanıcının profil "
        "sayfasında yer alan Şikayet Et düğmesini kullanarak iletebilirsiniz. Toplu şikayet "
        "bildiriminiz ayrıca kayıt altına alınmıştır ve moderatörler tarafından incelenecektir."
    ),
}


HUMAN_REVIEW_LABELS = {
    "ban_ceza_itirazi",
    "giris_hesap_erisim",
    "hesap_islemleri",
    "hile_itirazi",
    "teknik_uygulama_sorunu",
}


def build_auto_reply(label: str, confidence: float | None = None) -> dict[str, object]:
    reply = REPLY_TEMPLATES.get(label, FALLBACK_REPLY)
    return {
        "reply": reply,
        "requiresHumanReview": label in HUMAN_REVIEW_LABELS or (
            confidence is not None and confidence < HUMAN_REVIEW_CONFIDENCE_THRESHOLD
        ),
        "hasTemplate": label in REPLY_TEMPLATES,
    }
