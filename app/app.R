library(shiny)
library(shinyjs)
library(shinycssloaders)
library(bslib)
library(dplyr)
library(DALEX)
library(ggplot2)
library(xgboost)
library(recipes)
library(shiny.i18n)
library(curl)
library(showtext)
library(ragg)

options(shiny.useragg = TRUE)

font_add_google("Noto Sans", "noto_latin")
showtext_auto(enable = TRUE)
showtext_opts(dpi = 96)

i18n <- Translator$new(translation_json_path = "translation.json")

JEZYKI_CHOICES <- c(
  "🇵🇱 Polski"    = "pl",
  "🇬🇧 English"   = "en",
  "🇩🇪 Deutsch"   = "de",
  "🇪🇸 Español"   = "es",
  "🇫🇷 Français"  = "fr",
  "🇮🇹 Italiano"  = "it",
  "🇵🇹 Português" = "pt"
)

slownik_kolumn <- c(
  "make"                = "Marka",
  "model"               = "Model",
  "year"                = "Rok produkcji",
  "miles"               = "Przebieg [mile]",
  "engine_vol"          = "Pojemność silnika [l]",
  "engine_kW"           = "Moc silnika [kW]",
  "body_type"           = "Typ nadwozia",
  "transmission"        = "Skrzynia biegów",
  "fuel_type"           = "Rodzaj paliwa",
  "ulez"                = "Strefa czystego transportu (ULEZ)",
  "full_service"        = "Pełna historia serwisowa",
  "part_service"        = "Częściowa historia serwisowa",
  "part_warranty"       = "Częściowa gwarancja",
  "full_dealership"     = "Serwisowany w ASO",
  "first_year_road_tax" = "Podatek drogowy za 1. rok",
  "brand_new"           = "Fabrycznie nowy",
  "finance_available"   = "Dostępne finansowanie",
  "seller_type"         = "Typ sprzedawcy pojazdu",
  "damage_category"     = "Kategoria uszkodzeń",
  "sunroof"             = "Szyberdach",
  "leather"             = "Skórzane wnętrze",
  "heated_seats"        = "Podgrzewane siedzenia",
  "camera"              = "Kamera",
  "ac"                  = "Klimatyzacja",
  "alloy_rims"          = "Aluminiowe felgi",
  "sat_nav"             = "Nawigacja satelitarna",
  "smartphone_link"     = "Integracja ze smartfonem",
  "premium_lights"      = "Światła laserowe/matrycowe",
  "virtual_cockpit"     = "Cyfrowy wyświetlacz kierowcy",
  "premium_audio"       = "System audio klasy premium",
  "has_4wd"             = "Napęd na cztery koła",
  "tow_bar"             = "Hak holowniczy",
  "privacy_glass"       = "Przyciemniane szyby",
  "keyless_entry"       = "Dostęp bezkluczykowy",
  "style_pack"          = "Pakiet stylistyczny",
  "head_up_display"     = "Wyświetlacz Head-Up"
)

KOLUMNY_BINARNE <- c(
  "ulez", "full_service", "part_service", "part_warranty", "full_dealership",
  "brand_new", "finance_available", "sunroof", "leather", "heated_seats",
  "camera", "ac", "alloy_rims", "sat_nav", "smartphone_link", "premium_lights",
  "virtual_cockpit", "premium_audio", "has_4wd", "tow_bar", "privacy_glass",
  "keyless_entry", "style_pack", "head_up_display"
)

get_other_label <- function(col_name, is_binary, i18n) {
  raw_txt <- if (isTRUE(is_binary)) {
    i18n$t("brak informacji")
  } else {
    i18n$t("inne")
  }
  return(tolower(gsub("[\r\n]+", " ", trimws(raw_txt))))
}

granice_modelu <- list(
  year       = list(min = 2000, max = 2022),
  miles      = list(min = 0,    max = 210000),
  engine_kW  = list(min = 30,   max = 500),
  engine_vol = list(min = 0,    max = 6.5)
)

train_sample_3k <- readRDS("dane/train_sample_3k.rds")
model <- readRDS("model/final_xgboost_model.rds")
recipe_prepped_boosting <- readRDS("dane/recipe_prepped_boosting.rds")
slownik <- readRDS("dane/slownik.rds")

get_retained_levels <- function(prepped_recipe) {
  out <- list()
  tid_all <- tryCatch(recipes::tidy(prepped_recipe), error = function(e) NULL)
  if (is.null(tid_all) || !all(c("number", "type") %in% names(tid_all))) return(out)
  
  for (n in tid_all$number[tid_all$type == "other"]) {
    tr <- tryCatch(recipes::tidy(prepped_recipe, number = n), error = function(e) NULL)
    if (is.null(tr) || !all(c("terms", "retained") %in% names(tr))) next
    for (term in unique(as.character(tr$terms))) {
      out[[term]] <- unique(c(out[[term]], as.character(tr$retained[tr$terms == term])))
    }
  }
  out
}

retained_levels <- get_retained_levels(recipe_prepped_boosting)

# Poziomy specjalne dodawane przez step_novel()/step_unknown() - interfejs ma
# własną opcję "inne", więc nie pokazujemy ich jako zwykłych wartości.
POZIOMY_SPECJALNE <- c("inne", "nieznane")

# Zwraca posortowane opcje dla kolumny: z receptury, a gdy receptura nie
# zawiera tej kolumny (lub nie udało się jej odczytać) - z wartości w próbce.
get_choice_levels <- function(col_name, sample_vec) {
  lv <- retained_levels[[col_name]]
  base <- if (!is.null(lv) && length(lv) > 0) lv else as.character(na.omit(sample_vec))
  sort(setdiff(unique(as.character(base)), POZIOMY_SPECJALNE))
}

wszystkie_marki <- get_choice_levels("make", train_sample_3k$make)

# Modele przypisane do marek: słownik przefiltrowany do marek i modeli, które
# receptura rozróżnia (model z < 10 wystąpieniami i tak zostałby zamieniony na
# "inne", więc wybór go z listy dawałby mylące wrażenie).
wszystkie_modele <- get_choice_levels("model", train_sample_3k$model)
model_ok <- if (!is.null(retained_levels[["model"]])) {
  as.character(slownik$model) %in% wszystkie_modele
} else {
  rep(TRUE, nrow(slownik))
}
slownik_ui <- slownik[as.character(slownik$make) %in% wszystkie_marki & model_ok, , drop = FALSE]
modele_wg_marki <- lapply(
  split(as.character(slownik_ui$model), as.character(slownik_ui$make)),
  function(x) sort(unique(x))
)

# Lista modeli dla danej marki (zawsze z opcją "inne" na końcu).
get_models_for_make <- function(make) {
  if (is.null(make) || length(make) == 0 || is.na(make) || identical(make, "inne")) {
    return("inne")
  }
  mods <- modele_wg_marki[[as.character(make)]]
  unique(c(if (is.null(mods)) character(0) else mods, "inne"))
}

message(sprintf(
  "[UI] Marki: %d (źródło: %s), modele w słowniku UI: %d",
  length(wszystkie_marki),
  if (!is.null(retained_levels[["make"]])) "receptura" else "próbka (receptura nie zawiera 'make')",
  length(unique(unlist(modele_wg_marki)))
))
marki_bez_modeli <- setdiff(wszystkie_marki, names(modele_wg_marki))
if (length(marki_bez_modeli) > 0) {
  message("[UI] Marki bez modeli w słowniku (dostępna będzie tylko opcja 'inne'): ",
          paste(marki_bez_modeli, collapse = ", "))
}

# Powiązanie pojemności silnika z rodzajem paliwa: dla auta elektrycznego
# pojemność jest zablokowana na 0, dla pozostałych musi być większa od 0.
PALIWA_ELEKTRYCZNE <- c("elektryczny", "electric")
is_electric_fuel <- function(x) {
  length(x) == 1 && !is.na(x) && tolower(trimws(as.character(x))) %in% PALIWA_ELEKTRYCZNE
}
HAS_FUEL_RULE <- all(c("fuel_type", "engine_vol") %in% names(train_sample_3k))
if (HAS_FUEL_RULE &&
    !any(tolower(get_choice_levels("fuel_type", train_sample_3k$fuel_type)) %in% PALIWA_ELEKTRYCZNE)) {
  warning("[UI] W kolumnie fuel_type nie znaleziono poziomu 'elektryczny'/'electric' - ",
          "reguła pojemności silnika dla aut elektrycznych nie będzie działać. ",
          "Dopisz właściwą nazwę do PALIWA_ELEKTRYCZNE.", call. = FALSE)
}

custom_splits <- list(
  year = seq(2000, 2022, by = 1),
  miles = seq(0, 210000, by = 1000),
  engine_kW = seq(30, 500, by = 1),
  engine_vol = seq(0, 6.5, by = 0.1)
)

explainer_xgboost <- DALEX::explain(
  model = model,
  data = train_sample_3k,
  y = NULL,
  predict_function = function(model, data) {
    data_baked <- bake(recipe_prepped_boosting, data)
    dmat <- xgb.DMatrix(data = data_baked)
    preds <- predict(model, newdata = dmat)
    return(exp(preds))
  },
  label = "XGBoost",
  verbose = FALSE
)

ciemny_motyw <- bs_theme(
  version = 5,
  bootswatch = "darkly", 
  bg = "#1e1e2f",        
  fg = "#ffffff",
  primary = "#00d2ff"
)

ui <- page_sidebar(
  title = uiOutput("app_title"),
  window_title = "Asystent Wyceny Samochodów",
  theme = ciemny_motyw,
  
  tags$head(
    tags$script(HTML("
      // Serwer dostaje prośbę o uzupełnienie dopiero po utracie focusu.
      // Dzięki temu Backspace nie przerywa normalnego wpisywania nowej wartości.
      function requestDefaultRestore(id, type) {
        if (id) {
          Shiny.setInputValue(
            'restore_empty_field',
            { id: id, type: type, nonce: Date.now() },
            { priority: 'event' }
          );
        }
      }

      $(document).on('focusout', '#dynamic_inputs input[type=\"number\"]', function() {
        if ($(this).val() === '' || $(this).val() === null) {
          requestDefaultRestore(this.id, 'numeric');
        }
      });

      // Dla selectize sprawdzamy faktycznie wybraną wartość, a nie tekst
      // wpisany chwilowo do wyszukiwarki listy.
      $(document).on('focusout', '#dynamic_inputs .selectize-input input', function() {
        var $select = $(this).closest('.shiny-input-container').find('select');
        if ($select.length && (!$select.val() || $select.val().length === 0)) {
          requestDefaultRestore($select.attr('id'), 'categorical');
        }
      });

      // Wyszukiwanie we wszystkich listach rozwijanych Selectize — także w
      // wyborze języka. Po samym uzyskaniu fokusu pokazujemy pełne pole
      // wyszukiwania zamiast wcześniej wybranej etykiety.
      function selectizeOf(el) {
        var sel = $(el).closest('.shiny-input-container').find('select.selectized').get(0);
        return (sel && sel.selectize) ? sel.selectize : null;
      }

      // W standardowym Selectize close() wywołuje hideInput(), które czyści
      // tekst wyszukiwania. Gdy użytkownik nadal pisze, zachowujemy widoczne
      // i aktywne pole, aby mógł dopisać dowolną liczbę kolejnych znaków.
      function protectSearchInput(s) {
        if (!s || s.searchInputProtected) { return; }
        var originalHideInput = s.hideInput;
        s.hideInput = function() {
          var input = this.$control_input[0];
          if (this.$control.hasClass('sel-typing') && document.activeElement === input) {
            this.$control_input.css({ opacity: 1, position: 'relative', left: 0 });
            this.isInputHidden = false;
            return;
          }
          return originalHideInput.apply(this, arguments);
        };

        // Brak wyników bez migania. Gdy po wpisaniu znaku nic nie pasuje,
        // Selectize wywołuje w refreshOptions() close(), które chowa panel
        // (i czyści wpisany tekst), a komunikat musiałby dopiero potem
        // wrócić - stąd chwilowe znikanie pola. Zamiast tego, podczas
        // odświeżania z wpisaną frazą blokujemy to zamknięcie i od razu, w tym
        // samym wywołaniu (przed ponownym rysowaniem strony), wstawiamy
        // komunikat. Kolejny znak bez wyników nie zmienia więc widocznego
        // stanu, a znak dający wyniki normalnie odświeża listę.
        var originalClose = s.close;
        s.close = function() {
          if (this.keepOpenForNoResults) { return; }
          return originalClose.apply(this, arguments);
        };

        var originalRefreshOptions = s.refreshOptions;
        s.refreshOptions = function() {
          var query = this.$control_input.val() || '';
          var focused = document.activeElement === this.$control_input[0];
          this.keepOpenForNoResults = (query !== '' && focused);
          var result;
          try {
            result = originalRefreshOptions.apply(this, arguments);
          } finally {
            this.keepOpenForNoResults = false;
          }
          if (query !== '' && focused &&
              this.$dropdown_content.find('[data-selectable]').length === 0) {
            this.$control.addClass('sel-typing');
            this.$dropdown_content.html(
              $('<div>', { 'class': 'selectize-no-results', text: noResultsLabel() })
            );
            this.$dropdown.css({ display: 'block', visibility: 'visible' });
            this.positionDropdown();
          }
          return result;
        };
        s.searchInputProtected = true;
      }

      $(document).on('focusin', '.selectize-input input', function() {
        var s = selectizeOf(this);
        if (!s) { return; }
        protectSearchInput(s);
        s.$control.addClass('sel-typing');
        s.showInput();
        if (!s.isOpen) {
          s.open();
        }
      });

      // W pojedynczych listach Selectize wcześniejszy wybór powoduje stan
      // pełnego pola i biblioteka blokuje znaki drukowalne. Zatrzymujemy wyłącznie
      // jej obsługę keydown, pozostawiając domyślną akcję przeglądarki, dzięki
      // czemu znak trafia do widocznego pola input.
      document.addEventListener('keydown', function(event) {
        if (!event.target.matches('.selectize-input input')) {
          return;
        }
        if (event.ctrlKey || event.metaKey || event.altKey || !event.key || event.key.length !== 1) {
          return;
        }
        var s = selectizeOf(event.target);
        if (!s || (event.key === ' ' && s.$control_input.val() === '')) {
          return;
        }
        event.stopPropagation();
        s.showInput();
        if (!s.isOpen) {
          s.open();
        }
      }, true);

      function noResultsLabel() {
        var labels = {
          pl: 'Brak pasujących opcji',
          en: 'No matching options',
          de: 'Keine passenden Optionen',
          es: 'No hay opciones coincidentes',
          fr: 'Aucune option correspondante',
          it: 'Nessuna opzione corrispondente',
          pt: 'Nenhuma opção correspondente'
        };
        return labels[$('#selected_language').val()] || labels.en;
      }

      // Po każdej zmianie tekstu ustawiamy aktywną (podświetloną) opcję na
      // pierwsze dopasowanie. Domyślnie Selectize podświetlałby aktualnie
      // wybraną wartość, jeśli też pasuje do wpisanego tekstu.
      function handleSelectizeTextInput(input) {
        var s = selectizeOf(input);
        if (!s) { return; }
        var q = s.$control_input.val() || '';
        s.$control.toggleClass('sel-typing', q !== '' || document.activeElement === input);
        var $input = $(input);
        if (q !== ($input.data('lastq') || '')) {
          $input.data('lastq', q);
          if (q !== '') {
            window.setTimeout(function() {
              var $first = s.$dropdown_content.find('[data-selectable]').first();
              if ($first.length) {
                s.setActiveOption($first, true);
              }
            }, 0);
          }
        }
      }

      // Natywne nasłuchiwanie w fazie capture działa również wtedy, gdy
      // wewnętrzny kod Selectize zatrzymuje propagację zdarzenia.
      document.addEventListener('input', function(event) {
        if (event.target.matches('.selectize-input input')) {
          handleSelectizeTextInput(event.target);
        }
      }, true);

      // Po utracie focusu lub wyborze wartości wracamy do zwykłego wyglądu pola.
      function resetTyping(s) {
        if (!s) { return; }
        s.$control.removeClass('sel-typing');
        $(s.$control_input).data('lastq', '');
        if (s.$dropdown_content.find('.selectize-no-results').length) {
          s.$dropdown.hide();
        }
      }
      $(document).on('focusout', '.selectize-input input', function() {
        resetTyping(selectizeOf(this));
      });
      $(document).on('change', 'select.selectized', function() {
        resetTyping(this.selectize);
      });

      // Selectize domyślnie zamyka listę po Tab. Przechodzimy zamiast tego do
      // następnego pola formularza (po marce: bezpośrednio do modelu).
      document.addEventListener('keydown', function(event) {
        if (event.key !== 'Tab' || event.shiftKey || !event.target.matches('.selectize-input input')) {
          return;
        }

        var $currentGroup = $(event.target).closest('.shiny-input-container');

        // Jeśli użytkownik wpisał tekst i podświetlone jest dopasowanie, Tab
        // zatwierdza to dopasowanie. Dotyczy WSZYSTKICH list rozwijanych,
        // także wyboru języka. Dalszy ruch fokusu zależy od listy: pola
        // konfiguracji pojazdu obsługujemy poniżej, a pozostałe (np. język)
        // przechodzą dalej domyślnie przez przeglądarkę.
        var currentSelectize = selectizeOf(event.target);
        if (currentSelectize && currentSelectize.isOpen && currentSelectize.$activeOption &&
            currentSelectize.$control_input.val() !== '') {
          var pickedValue = currentSelectize.$activeOption.attr('data-value');
          if (pickedValue !== undefined) {
            currentSelectize.addItem(pickedValue);
          }
        }

        if (!event.target.matches('#dynamic_inputs .selectize-input input')) {
          return;
        }

        // Pomijamy zablokowane pola (np. pojemność silnika dla auta elektrycznego).
        var $nextGroup = $currentGroup.nextAll('.shiny-input-container').filter(function() {
          return $(this).find('input:disabled').length === 0;
        }).first();
        if (!$nextGroup.length) {
          return;
        }

        event.preventDefault();
        event.stopPropagation();

        var nextSelect = $nextGroup.find('select.selectized').get(0);
        if (nextSelect && nextSelect.selectize) {
          nextSelect.selectize.focus();
          nextSelect.selectize.open();
          if ($currentGroup.find('select').attr('id') === 'make') {
            Shiny.setInputValue('focus_model_after_make', Date.now(), { priority: 'event' });
          }
          return;
        }

        $nextGroup.find('input[type=\"number\"]').first().focus();
      }, true);

      // Aktualizacja modeli po zmianie marki jest asynchroniczna. Po jej
      // zakończeniu ponownie ustawiamy fokus na liście Model.
      Shiny.addCustomMessageHandler('focus-selectize', function(inputId) {
        window.setTimeout(function() {
          var select = document.getElementById(inputId);
          if (select && select.selectize) {
            select.selectize.focus();
            select.selectize.open();
          }
        }, 50);
      });
    
      window.addEventListener('scroll', function() {
        if (window.scrollY !== 0 || window.scrollX !== 0) {
          window.scrollTo(0, 0);
        }
      }, { passive: false });

      $(document).on('focusin', 'aside.sidebar input', function() {
        var el = this;
        var $wrapper = $(el).closest('.form-group');
        if ($wrapper.length) {
          el = $wrapper[0];
        }
        requestAnimationFrame(function() {
          el.scrollIntoView({ block: 'center', behavior: 'smooth' });
        });
      });

      $(document).on('focus', 'input[type=\"number\"]', function() {
        $(this).select();
      });

      $(document).on('keydown', 'aside.sidebar input, aside.sidebar .selectize-input input', function(e) {
        if (e.key === 'Enter') {
          e.preventDefault();
          var $focusable = $('aside.sidebar').find('input, select, textarea, [tabindex=\"0\"], .selectize-input input').filter(':visible').not(':disabled');
          var index = $focusable.index(this);
          if (index > -1 && index < $focusable.length - 1) {
            $focusable.eq(index + 1).focus();
          }
        }
      });

      $(document).on('shiny:sessioninitialized', function() {
        var fixTabsAccessibility = function() {
          $('.nav-underline .nav-link').each(function() {
            $(this).attr('tabindex', '0');
            if (!$(this).attr('role')) {
              $(this).attr('role', 'tab');
            }
          });
        };
        fixTabsAccessibility();
        $(document).on('shown.bs.tab', fixTabsAccessibility);
      });
      
      $(document).on('keydown', '.nav-underline .nav-link', function(e) {
        if (e.key === ' ' || e.key === 'Enter') {
          e.preventDefault();
          $(this).tab('show');
        }
      });
    ")),
    tags$style(HTML("
      html, body {
        overflow: hidden !important;
        height: 100% !important;
        width: 100% !important;
        position: fixed;
      }
      
      .form-group label {
        margin-bottom: 4px !important;
      }
      
      .form-group {
        margin-bottom: 12px !important; 
      }
      
      .bslib-page-sidebar, .bslib-sidebar-layout {
        height: 100vh !important;
        overflow: hidden !important;
      }
      
      aside.sidebar {
        max-width: 600px !important;
        width: var(--_sidebar-width, 290px) !important;
        flex-shrink: 0 !important;
        overflow: hidden !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
      }
      
      aside.sidebar:not([data-collapsed='true']):not(.collapsed) {
        min-width: 290px !important;
      }
      
      aside.sidebar .sidebar-content, 
      aside.sidebar > .well, 
      aside.sidebar > div:not(.collapse-toggle):not(.sidebar-toggle) {
        height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 1rem !important;
        box-sizing: border-box !important;
        width: 100% !important;
      }

      bslib-layout-sidebar:has(#dynamic_inputs),
      .bslib-sidebar-layout:has(#dynamic_inputs) {
        grid-template-columns: minmax(290px, min(600px, var(--_sidebar-width, 290px))) 1fr !important;
      }
      
      .form-control:focus, .selectize-input.focus {
        outline: none !important;
        border-color: #00d2ff !important;
        box-shadow: 0 0 8px rgba(0, 210, 255, 0.5) !important;
      }
      
      .custom-invalid .form-control, .form-control.custom-invalid, .form-group.custom-invalid .form-control {
        border-color: #ff6b6b !important;
        box-shadow: 0 0 0 0.25rem rgba(255, 107, 107, 0.25) !important;
        background-color: rgba(255, 107, 107, 0.05) !important;
      }
      
      .collapse-toggle, .sidebar-toggle, .bslib-sidebar-toggle, .sidebar-toggle-btn {
        display: none !important;
      }
      
      .card.bslib-value-box .value-box-area { padding: 10px !important; }
      .card.bslib-value-box .card-body { padding: 0 !important; }

      .navbar-brand, .sidebar-title-custom { font-size: 5rem !important; font-weight: 700 !important; color: #ffffff !important; }

      /* Nagłówek: tytuł po lewej, adnotacja o danych po prawej (na tej samej
         linii bazowej). Kontener rozciągamy na całą szerokość nagłówka, a przy
         braku miejsca (wąskie okno, dłuższe tłumaczenie) adnotacja sama
         przechodzi pod tytuł dzięki flex-wrap. */
      .navbar-brand {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        margin-right: 0 !important;
        white-space: normal !important;
      }

      #app_title {
        display: block;
        width: 100%;
      }

      .app-header-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: baseline;
        column-gap: 24px;
        row-gap: 0;
        width: 100%;
      }

      .app-header-note {
        font-size: 0.9rem !important;
        font-weight: 400 !important;
        color: #aab0cf !important;
        text-align: right;
      }

      @media (max-width: 576px) {
        .app-header-note {
          font-size: 0.8rem !important;
        }
      }
      .sidebar-section-title { text-align: center !important; width: 100% !important; display: block !important; font-size: 1.25rem !important; font-weight: 600 !important; margin-bottom: 15px !important; color: #00d2ff !important; }
      .plot-container { background-color: #ffffff !important; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
      .value-box-center { text-align: center !important; display: flex !important; flex-direction: column !important; justify-content: center !important; align-items: center !important; width: 100% !important; }
      
      .value-box-title { font-size: 1.1rem !important; font-weight: bold; display: block !important; margin-bottom: 0px !important; }
      .value-box-value { font-size: 2.8rem !important; color: #00d2ff !important; font-weight: 800; display: block !important; margin-bottom: 2px !important; line-height: 1.0 !important; }
      
      .card.bslib-card { margin-top: 0px !important; margin-bottom: 13px !important; border-radius: 8px !important; }
      .card-header { padding-top: 4px !important; padding-bottom: 4px !important; }
      .card-body { padding: 8px !important; }
      .row { margin-left: 0 !important; margin-right: 0 !important; }
      .col-12, .col-sm-12 { padding-left: 0 !important; padding-right: 0 !important; }

      .shiny-plot-output { position: relative !important; }
      /* Zachowuje ostatni poprawny wykres widoczny podczas jego odświeżania. */
      .shiny-plot-output.recalculating { opacity: 1 !important; }

      .btn-primary {
        background-color: #00d2ff !important;
        border-color: #00d2ff !important;
        color: #1e1e2f !important;
        font-weight: bold !important;
      }

      .btn-primary:hover, .btn-primary:focus {
        background-color: #ff4d4d !important;
        border-color: #ff4d4d !important;
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(255, 77, 77, 0.7) !important;
      }

      .btn-primary:disabled, 
      .btn-primary.disabled,
      button[id=\"predict\"][disabled],
      button[id=\"predict\"].disabled {
        background-color: #424666 !important;
        border-color: #424666 !important;
        color: #8c92b3 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        opacity: 0.65 !important;
      }

      .form-control, 
      .selectize-input {
        background-color: #1e1e2f !important;
        border: 1px solid #424666 !important;
        color: #00d2ff !important;
        border-radius: 6px !important;
      }

      .selectize-input input {
        color: #00d2ff !important;
      }

      input.form-control:disabled {
        background-color: #2a2c42 !important;
        color: #8c92b3 !important;
        border-color: #424666 !important;
        cursor: not-allowed !important;
        opacity: 0.7 !important;
      }

      .selectize-dropdown {
        background-color: #27293d !important;
        border: 1px solid #00d2ff !important;
        border-radius: 6px !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.5) !important;
        position: absolute !important;
        max-height: 200px !important;
        z-index: 1000 !important;
      }

      .selectize-dropdown .option {
        color: #ffffff !important;
        padding: 8px 12px !important;
        background-color: transparent !important;
      }

      .selectize-dropdown .option:hover,
      .selectize-dropdown .option.active {
        background-color: #00d2ff !important;
        color: #1e1e2f !important;
        font-weight: bold !important;
      }

      .selectize-dropdown .selectize-no-results {
        color: #ffffff !important;
        padding: 8px 12px !important;
        font-style: italic;
      }
      
      .plot-container, #ui_plot_bd, #ui_plot_cp, #plot_bd, #plot_cp {
        overflow: hidden !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
      }
      
      .bslib-sidebar-layout .main {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 0 17px 8px 17px !important;
      }
      
      .sidebar-menu, .bslib-sidebar {
        overflow-x: hidden !important;
        white-space: nowrap !important;
      }

      .selectize-input.has-items > input {
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
      }

      /* Po rozpoczęciu pisania (klasa sel-typing ustawiana w JS) pokazujemy
         wpisywany tekst zamiast wybranej wartości, aby wyszukiwanie działało
         od razu po focusie, bez konieczności kasowania wyboru Backspace'em. */
      .selectize-input.sel-typing > input {
        position: static !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        /* Selectize nadaje temu polu w stylu inline szerokość rzędu kilku
           pikseli. Podczas wyszukiwania musi ono wypełniać kontrolkę, aby
           cały wpisywany tekst pozostał czytelny. */
        width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
      }

      .selectize-input.sel-typing > .item {
        display: none !important;
      }

      .selectize-input .item {
        white-space: normal !important;
        word-break: break-word !important;
        display: inline-block !important;
        max-width: calc(100% - 15px) !important;
      }

      .selectize-input {
        height: auto !important;
        padding-bottom: 6px !important;
      }
    "))
  ),
  
  usei18n(i18n),
  shinyjs::useShinyjs(),
  
  sidebar = sidebar(
    id = "sidebar_panel", 
    tags$label(
      class = "control-label",
      `for` = "selected_language",
      textOutput("language_label", inline = TRUE)
    ),
    selectInput(
      inputId = "selected_language",
      label = NULL,
      choices = as.list(JEZYKI_CHOICES),
      selected = "pl"
    ),
    hr(style = "margin-top: 5px; margin-bottom: 15px; border-color: #424666;"),
    
    uiOutput("sidebar_title"),
    bg = "#27293d", 
    uiOutput("dynamic_inputs"),
    hr(),
    uiOutput("predict_button_ui"),
    uiOutput("validation_msg")
  ),
  
  fluidPage(
    style = "padding: 0px !important;",
    div(
      class = "row g-0",
      style = "margin-top: 0px !important; margin-bottom: 4px !important;",
      column(12, uiOutput("price_box"))
    ),
    div(
      class = "row g-0",
      column(12,
             navset_card_underline(
               id = "main_tabs",
               nav_panel(title = uiOutput("tab_bd_title"), uiOutput("ui_plot_bd")),
               nav_panel(title = uiOutput("tab_cp_title"), uiOutput("ui_plot_cp"))
             )
      )
    )
  )
)

server <- function(input, output, session) {
  
  default_make <- wszystkie_marki[1]
  has_predicted <- reactiveVal(FALSE)
  prediction_request <- reactiveVal(0)
  last_predict_click <- reactiveVal(0)
  last_engine_vol <- reactiveVal(NULL)
  
  numeric_cols <- names(train_sample_3k)[sapply(train_sample_3k, is.numeric)]
  
  get_range <- function(col_name) {
    min_val <- if (col_name %in% names(granice_modelu)) granice_modelu[[col_name]]$min else min(train_sample_3k[[col_name]], na.rm = TRUE)
    max_val <- if (col_name %in% names(granice_modelu)) granice_modelu[[col_name]]$max else max(train_sample_3k[[col_name]], na.rm = TRUE)
    if (col_name %in% c("engine_vol", "miles")) min_val <- 0
    list(min = min_val, max = max_val)
  }
  
  is_value_invalid <- function(val, min_val, max_val) {
    if (is.null(val) || length(val) == 0 || is.na(val) || identical(val, "")) return(TRUE)
    val_num <- suppressWarnings(as.numeric(val))
    is.na(val_num) || isTRUE(val_num > max_val) || isTRUE(val_num < min_val)
  }
  
  # Wartość domyślna pola liczbowego (mediana z próbki). Dla pojemności
  # silnika liczona tylko z wartości dodatnich, bo 0 jest dopuszczalne
  # wyłącznie dla aut elektrycznych.
  get_default_numeric <- function(col_name) {
    vec <- train_sample_3k[[col_name]]
    if (identical(col_name, "engine_vol")) vec <- vec[!is.na(vec) & vec > 0]
    digits_round <- if (col_name %in% c("miles", "year", "engine_kW")) 0 else 2
    round(median(vec, na.rm = TRUE), digits_round)
  }
  
  # Poprawność wartości jednego pola liczbowego. Pojemność silnika zależy od
  # rodzaju paliwa: dla elektrycznego pole jest zablokowane na 0 (traktujemy je
  # jako poprawne), dla pozostałych paliw wartość musi być większa od 0.
  is_field_invalid <- function(col_name, val, fuel_value) {
    rng <- get_range(col_name)
    if (HAS_FUEL_RULE && identical(col_name, "engine_vol")) {
      if (is_electric_fuel(fuel_value)) return(FALSE)
      num <- suppressWarnings(as.numeric(val))
      return(isTRUE(is_value_invalid(val, rng$min, rng$max)) || isTRUE(num <= 0))
    }
    is_value_invalid(val, rng$min, rng$max)
  }
  
  # Reaktywna, NIEizolowana lista - przelicza się natychmiast przy każdej
  # zmianie dowolnego pola liczbowego lub rodzaju paliwa (żywsza walidacja), i
  # to jest jedyne miejsce, które faktycznie sprawdza poprawność wartości.
  invalid_fields <- reactive({
    fuel <- if (HAS_FUEL_RULE) input$fuel_type else NULL
    setNames(
      lapply(numeric_cols, function(col_name) {
        is_field_invalid(col_name, input[[col_name]], fuel)
      }),
      numeric_cols
    )
  })
  
  # Przywraca wartość wyłącznie na żądanie wysłane przez focusout w JavaScripcie.
  # To zapobiega resetowaniu pola podczas edycji klawiszem Backspace.
  observeEvent(input$restore_empty_field, {
    field <- input$restore_empty_field
    col_name <- field$id
    
    req(!is.null(col_name), col_name %in% names(train_sample_3k))
    
    if (is.numeric(train_sample_3k[[col_name]])) {
      default_value <- if (HAS_FUEL_RULE && identical(col_name, "engine_vol") &&
                           is_electric_fuel(input$fuel_type)) 0 else get_default_numeric(col_name)
      updateNumericInput(session, inputId = col_name, value = default_value)
    } else if (identical(col_name, "make")) {
      updateSelectizeInput(session, "make", selected = "inne")
    } else if (identical(col_name, "model")) {
      updateSelectizeInput(session, "model", selected = "inne")
    } else {
      # "inne" jest dostępną w każdej liście kategorią oznaczającą brak danych.
      updateSelectizeInput(session, col_name, selected = "inne")
    }
  }, ignoreInit = TRUE)
  
  # Zapamiętuje ostatnią dodatnią pojemność silnika, aby po zmianie paliwa z
  # elektrycznego na inne przywrócić wartość, którą użytkownik miał wcześniej.
  observeEvent(input$engine_vol, {
    v <- suppressWarnings(as.numeric(input$engine_vol))
    if (length(v) == 1 && !is.na(v) && v > 0) last_engine_vol(v)
  }, ignoreNULL = TRUE)
  
  # Paliwo elektryczne: pojemność silnika zablokowana i ustawiona na 0.
  # Pozostałe paliwa: pole odblokowane, a wartość musi być większa od 0 - jeśli
  # jest zerowa lub pusta, przywracamy ostatnią dodatnią (albo domyślną).
  # Po zmianie języka pole jest tworzone od nowa już z właściwym stanem
  # (patrz dynamic_inputs), więc ten obserwator reaguje tylko na zmianę paliwa.
  observeEvent(input$fuel_type, {
    req(HAS_FUEL_RULE)
    if (is_electric_fuel(input$fuel_type)) {
      updateNumericInput(session, "engine_vol", value = 0)
      shinyjs::disable("engine_vol")
    } else {
      shinyjs::enable("engine_vol")
      cur <- suppressWarnings(as.numeric(isolate(input$engine_vol)))
      if (length(cur) != 1 || is.na(cur) || cur <= 0) {
        restore_val <- last_engine_vol()
        if (is.null(restore_val)) restore_val <- get_default_numeric("engine_vol")
        updateNumericInput(session, "engine_vol", value = restore_val)
      }
    }
  }, ignoreNULL = TRUE)
  
  observeEvent(input$selected_language, {
    req(input$selected_language)
    update_lang(input$selected_language, session)
    # Przycisk predykcji jest renderowany ponownie z licznikiem od zera.
    # Reset po stronie serwera sprawia, że pierwsze kliknięcie po zmianie
    # języka jest traktowane jako nowe żądanie predykcji.
    last_predict_click(0)
  }, priority = 20)
  
  # Fokus na Model jest potrzebny tylko po użyciu Tab w polu Marka, nie po
  # początkowym zainicjalizowaniu formularza ani po zwykłym wyborze myszą.
  observeEvent(input$focus_model_after_make, {
    session$sendCustomMessage("focus-selectize", "model")
  }, ignoreInit = TRUE)
  
  current_lang <- reactive({
    req(input$selected_language)
    input$selected_language
  })
  
  # Nagłówek: tytuł po lewej, informacja o źródle danych po prawej, w tej samej
  # linii. Oba napisy są tłumaczone razem przy zmianie języka.
  output$app_title <- renderUI({
    current_lang()
    div(
      class = "app-header-row",
      tags$span(i18n$t("Asystent Wyceny Samochodów"), style = "font-size: 26px; font-weight: bold;"),
      tags$span(i18n$t("Dane pochodzą z UK, październik 2022"), class = "app-header-note")
    )
  })
  
  output$language_label <- renderText({
    current_lang()
    i18n$t("Język / Language:")
  })
  
  output$sidebar_title <- renderUI({ 
    current_lang()
    span(i18n$t("Konfiguracja pojazdu"), class = "sidebar-section-title") 
  })
  
  output$predict_button_ui <- renderUI({ 
    current_lang()
    btn <- actionButton("predict", i18n$t("Oblicz i Wyjaśnij"), class = "btn-primary w-100")
    if (isTRUE(isolate(validation_status()))) {
      btn <- tagAppendAttributes(btn, disabled = "disabled")
    }
    btn
  })
  
  # Włącza/wyłącza przycisk bez jego ponownego tworzenia (patrz komentarz wyżej).
  # Zależność od current_lang() dodatkowo potwierdza stan po przebudowie przycisku.
  observe({
    current_lang()
    if (isTRUE(validation_status())) {
      shinyjs::disable("predict")
    } else {
      shinyjs::enable("predict")
    }
  })
  
  output$tab_bd_title <- renderUI({ 
    current_lang()
    i18n$t("Lokalna dekompozycja (Break Down)") 
  })
  
  output$tab_cp_title <- renderUI({ 
    current_lang()
    i18n$t("Profil Ceteris Paribus") 
  })
  
  observeEvent(input$predict, {
    current_click <- input$predict
    previous_click <- last_predict_click()
    last_predict_click(current_click)
    
    # Ponowne utworzenie przycisku (np. po zmianie języka) resetuje jego
    # licznik do zera. Nie jest to żądanie nowej predykcji.
    if (is.null(current_click) || current_click <= previous_click) {
      return()
    }
    
    if (isTRUE(validation_status())) {
      return()
    }
    has_predicted(TRUE)
    prediction_request(prediction_request() + 1)
  }, ignoreInit = TRUE)
  
  # Generowanie pól - przerysowywane wyłącznie przy zmianie języka
  output$dynamic_inputs <- renderUI({
    current_lang() 
    
    cols <- names(train_sample_3k) 
    first_cols <- intersect(c("make", "model"), cols)
    remaining_cols <- setdiff(cols, first_cols)
    
    num_cols <- remaining_cols[sapply(train_sample_3k[remaining_cols], is.numeric)]
    cat_cols <- remaining_cols[!sapply(train_sample_3k[remaining_cols], is.numeric)]
    
    ordered_cols <- c(first_cols, num_cols, cat_cols)
    selectize_opts <- list(create = FALSE)
    valid_make_choices <- c(wszystkie_marki, "inne")
    
    input_list <- lapply(ordered_cols, function(col_name) {
      vec <- train_sample_3k[[col_name]]
      base_label <- if (col_name %in% names(slownik_kolumn)) slownik_kolumn[[col_name]] else col_name
      translated_col_name <- i18n$t(base_label)
      
      if (is.numeric(vec)) {
        rng <- get_range(col_name)
        min_val <- rng$min
        max_val <- rng$max
        
        med_val <- get_default_numeric(col_name)
        
        # Pojemność silnika dla paliwa elektrycznego: zablokowana i równa 0.
        # Stan ustalamy już przy tworzeniu pola, bo po zmianie języka pole
        # powstaje od nowa, a komunikat shinyjs mógłby trafić w stary element.
        engine_vol_locked <- HAS_FUEL_RULE && identical(col_name, "engine_vol") &&
          is_electric_fuel(isolate(input$fuel_type))
        
        # Pobieramy obecną wartość jeśli istnieje, w przeciwnym razie medianę
        current_val <- isolate(input[[col_name]])
        if (is.null(current_val) || is.na(current_val)) {
          current_val <- med_val
        }
        if (engine_vol_locked) current_val <- 0
        
        num_input <- numericInput(
          inputId = col_name,
          label = paste0(translated_col_name, " (", i18n$t("Zakres"), ": ", min_val, " - ", max_val, "):"),
          value = current_val,
          step = if(col_name %in% c("engine_vol")) 0.1 else 1
        )
        
        previous_val <- isolate(input[[col_name]])
        if (!is.null(previous_val) &&
            isTRUE(is_field_invalid(col_name, previous_val, isolate(input$fuel_type)))) {
          num_input$children[[2]] <- tagAppendAttributes(num_input$children[[2]], class = "custom-invalid")
        }
        
        # Wartość domyślna dostępna dla skryptu JS na polu i jego kontenerze.
        num_input$children[[2]] <- tagAppendAttributes(
          num_input$children[[2]],
          `data-default` = med_val
        )
        if (engine_vol_locked) {
          num_input$children[[2]] <- tagAppendAttributes(num_input$children[[2]], disabled = "disabled")
        }
        tagAppendAttributes(num_input, `data-default` = med_val)
        
      } else {
        # Opcje z receptury (wszystkie poziomy rozróżniane przez model), a nie
        # z próbki, która mogła pominąć rzadsze wartości.
        raw_vals <- get_choice_levels(col_name, vec)
        
        is_strictly_binary_vals <- length(raw_vals) > 0 && all(tolower(raw_vals) %in% c("0", "1", "true", "false", "tak", "nie"))
        is_binary_col <- (col_name %in% KOLUMNY_BINARNE) || is_strictly_binary_vals
        other_label <- get_other_label(col_name, is_binary_col, i18n)
        
        if (col_name == "make") {
          make_choices <- c(raw_vals, "inne")
          names(make_choices) <- c(raw_vals, other_label)
          
          current_val <- isolate(input[["make"]])
          if (is.null(current_val) || is.na(current_val) || !current_val %in% make_choices) {
            current_val <- default_make
          }
          
          selectizeInput("make", paste0(translated_col_name, ":"), choices = as.list(make_choices), selected = current_val, options = selectize_opts)
          
        } else if (col_name == "model") {
          current_make <- isolate(input[["make"]])
          if (is.null(current_make) || is.na(current_make) || !current_make %in% valid_make_choices) {
            current_make <- default_make
          }
          
          dostepne_modele <- get_models_for_make(current_make)
          model_choices <- dostepne_modele
          names(model_choices) <- ifelse(model_choices == "inne", other_label, model_choices)
          
          current_val <- isolate(input[["model"]])
          if (is.null(current_val) || is.na(current_val) || !current_val %in% dostepne_modele) current_val <- dostepne_modele[1]
          
          selectizeInput("model", paste0(translated_col_name, ":"), choices = as.list(model_choices), selected = current_val, options = selectize_opts)
          
        } else {
          choices_vec <- c(raw_vals, "inne")
          translated_names <- sapply(choices_vec, function(val) {
            res <- if (identical(val, "inne")) {
              other_label
            } else if (isTRUE(is_binary_col)) {
              if (val %in% c("1", "TRUE", "true", "Tak", "tak")) i18n$t("tak") else i18n$t("nie")
            } else {
              i18n$t(val)
            }
            res <- gsub("[\r\n]+", " ", trimws(res))
            if (!col_name %in% c("make", "model")) res <- tolower(res)
            return(res)
          })
          
          names(choices_vec) <- translated_names
          current_val <- isolate(input[[col_name]])
          if (is.null(current_val) || is.na(current_val) || !current_val %in% choices_vec) current_val <- choices_vec[1]
          
          selectizeInput(col_name, paste0(translated_col_name, ":"), choices = as.list(choices_vec), selected = current_val, options = selectize_opts)
        }
      }
    })
    do.call(tagList, input_list)
  })
  
  # Aktualizacja listy modeli po zmianie marki
  observeEvent(input$make, {
    req(input$make)
    current_model <- isolate(input$model)
    
    dostepne_modele <- get_models_for_make(input$make)
    
    other_label <- get_other_label("model", FALSE, i18n)
    model_choices <- dostepne_modele
    names(model_choices) <- ifelse(model_choices == "inne", other_label, model_choices)
    
    selected_val <- if (!is.null(current_model) && !is.na(current_model) && current_model %in% dostepne_modele) current_model else dostepne_modele[1]
    updateSelectizeInput(session, "model", choices = as.list(model_choices), selected = selected_val)
  }, ignoreInit = TRUE)
  
  # Sprawdzanie poprawności formularza - oparte o wspólne invalid_fields().
  validation_status <- reactive({
    any(unlist(invalid_fields()))
  })
  
  observe({
    current_lang()
    invalid <- invalid_fields()
    for (col_name in names(invalid)) {
      if (isTRUE(invalid[[col_name]])) {
        shinyjs::addClass(id = col_name, class = "custom-invalid")
      } else {
        shinyjs::removeClass(id = col_name, class = "custom-invalid")
      }
    }
  })
  
  output$validation_msg <- renderUI({
    current_lang()
    if (isTRUE(validation_status())) {
      div(style = "color: #ff6b6b; font-weight: bold; text-align: center; margin-top: 12px;",
          icon("exclamation-triangle"), paste0(" ", i18n$t("Wypełnij lub popraw zaznaczone pola przed obliczeniem!")))
    } else {
      NULL
    }
  })
  
  # Migawka formularza powstaje wyłącznie po kliknięciu przycisku. Zmiana
  # pojedynczego pola nie przelicza ceny ani wykresów.
  new_obs <- eventReactive(prediction_request(), {
    req(!validation_status())
    
    cols_ui <- names(train_sample_3k) 
    data_list <- lapply(cols_ui, function(col_name) {
      # Odczytujemy stan formularza w chwili kliknięcia przycisku.
      val <- input[[col_name]]
      
      # Auto elektryczne ma pojemność silnika 0, niezależnie od chwilowego
      # stanu pola (zablokowane pole mogło jeszcze nie zdążyć się zaktualizować).
      if (HAS_FUEL_RULE && identical(col_name, "engine_vol") && is_electric_fuel(input$fuel_type)) {
        val <- 0
      }
      
      if (col_name == "make" && (is.null(val) || length(val) == 0 || is.na(val))) {
        # Pusta marka oznacza brak konkretnej marki w danych wejściowych.
        val <- "inne"
      }
      
      if (is.null(val) || length(val) == 0 || (length(val) == 1 && is.na(val)) || val == "") {
        if (is.numeric(train_sample_3k[[col_name]])) {
          val <- get_default_numeric(col_name)
        } else {
          val <- as.character(train_sample_3k[[col_name]][1])
        }
      }
      return(as.character(val))
    })
    
    names(data_list) <- cols_ui
    df <- as.data.frame(data_list, stringsAsFactors = FALSE)
    
    for (col_name in names(train_sample_3k)) {
      if (is.numeric(train_sample_3k[[col_name]])) {
        df[[col_name]] <- as.numeric(df[[col_name]])
      } else if (is.factor(train_sample_3k[[col_name]])) {
        orig_levels <- levels(train_sample_3k[[col_name]])
        # Poziomy z próbki + wszystkie poziomy z receptury (oferowane w UI) +
        # "inne". Bez tego marka/model spoza poziomów próbki zamieniłaby się w
        # NA, a dalej w "nieznane", co po cichu zafałszowałoby predykcję.
        all_levels <- union(union(orig_levels, retained_levels[[col_name]]), "inne")
        df[[col_name]] <- factor(df[[col_name]], levels = all_levels)
      }
    }
    return(df)
  })
  
  # Wyniki modelu są liczone tylko po kliknięciu przycisku i pozostają dostępne
  # przy zmianie języka. Dzięki temu nie znikają ani nie wyliczają się ponownie.
  predicted_price <- eventReactive(prediction_request(), {
    req(!validation_status())
    tryCatch(
      predict(explainer_xgboost, newdata = new_obs()),
      error = function(e) 0
    )
  })
  
  breakdown_data <- eventReactive(prediction_request(), {
    req(!validation_status())
    set.seed(2027)
    predict_parts(
      explainer = explainer_xgboost,
      new_observation = new_obs(),
      type = "break_down",
      N = 100
    )
  })
  
  profile_data <- eventReactive(prediction_request(), {
    req(!validation_status())
    wybrane_zmienne <- intersect(c("year", "miles", "engine_kW", "engine_vol"), names(train_sample_3k))
    set.seed(2027)
    predict_profile(
      explainer = explainer_xgboost,
      new_observation = new_obs(),
      variables = wybrane_zmienne,
      variable_splits = custom_splits
    )
  })
  
  active_font_family <- reactive({ "noto_latin" })
  
  output$price_box <- renderUI({
    req(has_predicted())
    current_lang()
    prognozowana_cena <- predicted_price()
    cena_format <- paste0(format(round(prognozowana_cena, 0), big.mark = " "), " GBP")
    
    value_box(
      title = div(class = "value-box-center", 
                  span(i18n$t("Prognozowana Wartość Rynkowa Pojazdu:"), class = "value-box-title"),
                  span(cena_format, class = "value-box-value")
      ),
      value = NULL, theme = "secondary",
      styles = "background-color: #27293d !important; border: 1px solid #3d3f5d; padding: 4px 6px 0px 6px !important; margin: 0px !important;"
    )
  })
  
  output$ui_plot_bd <- renderUI({
    req(has_predicted())
    div(class = "plot-container", withSpinner(plotOutput("plot_bd"), type = 6, color = "#00d2ff", size = 1.2))
  })
  
  output$plot_bd <- renderPlot({
    req(has_predicted())
    current_lang()
    font_fam <- active_font_family()
    bd_auto <- breakdown_data()
    bd_auto <- bd_auto[bd_auto$variable_name != "residual", ]
    
    for (i in seq_len(nrow(bd_auto))) {
      var_name  <- trimws(as.character(bd_auto$variable_name[i]))
      var_val   <- trimws(as.character(bd_auto$variable_value[i]))
      orig_var  <- trimws(as.character(bd_auto$variable[i]))
      
      if (is.na(orig_var) || orig_var %in% c("intercept", "prediction", "+ all other factors") || is.na(var_name) || identical(var_name, "")) next
      
      col_label_pl <- if (var_name %in% names(slownik_kolumn)) slownik_kolumn[[var_name]] else var_name
      translated_col <- i18n$t(col_label_pl)
      
      is_num <- if (var_name %in% names(train_sample_3k)) is.numeric(train_sample_3k[[var_name]]) else FALSE
      # Kolumny liczbowe nigdy nie są binarne - inaczej np. pojemność silnika
      # równa 0 byłaby pokazana jako "nie" zamiast jako liczba 0.
      is_binary <- !isTRUE(is_num) &&
        ((var_name %in% KOLUMNY_BINARNE) || (tolower(var_val) %in% c("0", "1", "true", "false")))
      
      # sprawdzenie czy wartość oznacza brak danych ("inne", "NA", NA)
      is_missing_or_other <- is.na(var_val) || var_val %in% c("inne", "NA", "na", "")
      
      translated_val <- if (is_missing_or_other) {
        get_other_label(var_name, is_binary, i18n)
      } else if (isTRUE(is_binary)) {
        if (var_val %in% c("1", "TRUE", "true", "1.0", "Tak", "tak")) i18n$t("tak") else i18n$t("nie")
      } else if (isTRUE(is_num)) {
        var_val
      } else if (var_name %in% c("make", "model")) {
        var_val
      } else {
        i18n$t(var_val)
      }
      
      if (!var_name %in% c("make", "model") && !isTRUE(is_num)) translated_val <- tolower(translated_val)
      bd_auto$variable[i] <- paste0(translated_col, " = ", translated_val)
    }
    
    p <- plot(bd_auto, digits = 0, text_family = font_fam)
    for (i in seq_along(p$layers)) {
      if (!is.null(p$layers[[i]]$mapping$height)) {
        p$layers[[i]]$mapping$width <- p$layers[[i]]$mapping$height
        p$layers[[i]]$mapping$height <- NULL
      }
      if ("GeomText" %in% class(p$layers[[i]]$geom)) p$layers[[i]]$aes_params$size <- 5
    }
    
    p$coordinates$clip <- "off"
    res_plot <- p + theme_minimal(base_family = font_fam) +
      theme(
        strip.text = element_blank(), strip.background = element_blank(),
        plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
        axis.text.x = element_text(color = "black", size = 13), 
        axis.text.y = element_text(color = "black", size = 13),
        legend.position = "none", plot.margin = margin(10, 40, 10, 10, "pt")
      ) +
      labs(title = i18n$t("Dlaczego model tak wycenił ten konkretny samochód?"), x = i18n$t("Cena [GBP]"))
    
    suppressMessages(print(res_plot))
  })
  
  output$ui_plot_cp <- renderUI({ 
    req(has_predicted())
    div(class = "plot-container", withSpinner(plotOutput("plot_cp"), type = 6, color = "#00d2ff", size = 1.2))
  })
  
  output$plot_cp <- renderPlot({
    req(has_predicted())
    current_lang()
    font_fam <- active_font_family()
    cp_auto <- profile_data()
    
    tys_suffix <- paste0(" ", i18n$t("tys."))
    tys <- function(x) { 
      ifelse(!is.na(x) & abs(x) >= 5000, paste0(round(x / 1000), tys_suffix), as.character(round(x))) 
    }
    
    p <- suppressMessages(suppressWarnings(plot(cp_auto)))
    
    if (!is.null(p$data) && "_vname_" %in% names(p$data)) {
      p$data$`_vname_` <- as.character(p$data$`_vname_`)
      p$data$`_vname_` <- sapply(p$data$`_vname_`, function(v) {
        col_label_pl <- if (v %in% names(slownik_kolumn)) slownik_kolumn[[v]] else v
        i18n$t(col_label_pl)
      })
      p$data$`_vname_` <- factor(p$data$`_vname_`, levels = unique(p$data$`_vname_`))
    }
    
    for (i in seq_along(p$layers)) {
      if (is.data.frame(p$layers[[i]]$data) && "_vname_" %in% names(p$layers[[i]]$data)) {
        p$layers[[i]]$data$`_vname_` <- as.character(p$layers[[i]]$data$`_vname_`)
        p$layers[[i]]$data$`_vname_` <- sapply(p$layers[[i]]$data$`_vname_`, function(v) {
          col_label_pl <- if (v %in% names(slownik_kolumn)) slownik_kolumn[[v]] else v
          i18n$t(col_label_pl)
        })
        p$layers[[i]]$data$`_vname_` <- factor(p$layers[[i]]$data$`_vname_`, levels = unique(p$layers[[i]]$data$`_vname_`))
      }
    }
    
    p + theme_minimal(base_family = font_fam) +
      theme(
        plot.margin = margin(10, 15, 10, 15, "pt"),
        strip.text = element_text(face = "bold", size = 14),
        plot.title = element_text(face = "bold", size = 18, hjust = 0.5, margin = margin(b = 15)),
        strip.background = element_blank(),
        axis.text.x = element_text(color = "black", size = 12),
        axis.text.y = element_text(color = "black", size = 12),
        axis.title.x = element_text(color = "black", size = 14, margin = margin(t = 12)),
        axis.title.y = element_text(color = "black", size = 14, margin = margin(r = 12)),
        plot.subtitle = element_blank()
      ) +
      scale_y_continuous(labels = tys) +
      scale_x_continuous(labels = tys) +
      labs(
        title = i18n$t("Jak zmieniłaby się cena, jeśli zmieniono wartość tylko jednej zmiennej?"),
        x = i18n$t("Wartość zmiennej"),
        y = i18n$t("Prognozowana cena auta")
      )
  })
}

shinyApp(ui = ui, server = server)