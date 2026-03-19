"""
NewsCollectionAgent - агент для сбора новостей и медиа из множества источников:
HuggingFace (Gazeta), Kaggle (Lenta, Russian News), RSS, HTML-парсер
"""

import pandas as pd
import numpy as np
import yaml
import os
import re
import zipfile
import glob
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataCollectionAgent:
    """
    Агент для сбора и унификации новостей и медиа из различных источников.

    Поддерживаемые источники:
    - HuggingFace datasets (Gazeta)
    - Kaggle datasets (Lenta, Russian News 2020, Large Russian News)
    - RSS parser (Lenta, RIA, TASS, Kommersant)
    - HTML parser (скрапинг новостных сайтов)
    """

    def __init__(self, config_path: str = 'config.yaml'):
        self._project_root = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.sources = self.config.get('sources', [])
        self.output_schema = self.config.get('output_schema', {})
        self.target_size = self.config.get('target_size', 10000)

        self.collection_stats = {
            'total_collected': 0,
            'sources_used': [],
            'start_time': None,
            'end_time': None
        }

    def run(self, sources: Optional[List[Dict]] = None) -> pd.DataFrame:
        self.collection_stats['start_time'] = datetime.now()
        sources_to_use = sources if sources else self.sources

        print("Запуск NewsCollectionAgent")
        print(f"Целевой размер датасета: {self.target_size} записей")
        print(f"Количество источников: {len(sources_to_use)}\n")

        all_dataframes = []

        for idx, source in enumerate(sources_to_use, 1):
            source_type = source.get('type')
            print(f"[{idx}/{len(sources_to_use)}] Обработка источника: {source_type}")

            try:
                if source_type == 'hf_dataset':
                    df = self._load_hf_dataset(source)
                elif source_type == 'kaggle_dataset':
                    df = self._load_kaggle_dataset(source)
                elif source_type == 'rss_parser':
                    df = self._load_rss_data(source)
                elif source_type == 'html_parser':
                    df = self._load_html_data(source)
                else:
                    print(f"Неизвестный тип источника: {source_type}")
                    continue

                if df is not None and not df.empty:
                    all_dataframes.append(df)
                    self.collection_stats['sources_used'].append(source_type)
                    print(f"Получено {len(df)} записей из {source_type}\n")
                else:
                    print(f"Пустые данные от {source_type}, пропуск\n")

            except Exception as e:
                print(f"Ошибка при обработке {source_type}: {str(e)}\n")
                continue

        if not all_dataframes:
            raise ValueError("Не удалось собрать данные ни из одного источника")

        print("Объединение данных из всех источников...")
        unified_df = self._merge_sources(all_dataframes)

        if len(unified_df) > self.target_size:
            unified_df = unified_df.sample(n=self.target_size, random_state=42)
            print(f"Датасет уменьшен до {self.target_size} записей")
        else:
            print(f"Итого записей: {len(unified_df)} (цель: {self.target_size})")

        self.collection_stats['total_collected'] = len(unified_df)
        self.collection_stats['end_time'] = datetime.now()
        unified_df['collected_at'] = datetime.now()

        print("\nСбор завершен!")
        print(f"Итоговый размер: {len(unified_df)} записей")
        print(f"Время выполнения: {self.collection_stats['end_time'] - self.collection_stats['start_time']}")

        return unified_df

    def _load_hf_dataset(self, source: Dict) -> Optional[pd.DataFrame]:
        """Загрузка датасета из HuggingFace. Сохраняет в data/raw/hf_<name>/."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("Установите: pip install datasets")
            return None

        dataset_name = source.get('name', 'IlyaGusev/gazeta')
        limit = source.get('limit', 2000)
        safe_name = dataset_name.replace('/', '_')
        local_dir = f"data/raw/hf_{safe_name}"
        local_csv = os.path.join(local_dir, "data.csv")

        print(f"   Загрузка {dataset_name}...")

        df = None
        if os.path.exists(local_csv):
            df = pd.read_csv(local_csv)
            print(f"   Загружено из локального кэша {local_dir}")

        if df is None or df.empty:
            try:
                dataset = load_dataset(dataset_name, split='train')
                df = dataset.to_pandas()
                df = self._standardize_hf_columns(df, dataset_name)
                df['source'] = f"hf:{dataset_name}"
                os.makedirs(local_dir, exist_ok=True)
                df.to_csv(local_csv, index=False, encoding='utf-8')
                print(f"   Сохранено в {local_dir}")
            except Exception as e:
                print(f"   Ошибка загрузки HF: {e}")
                return None

        if len(df) > limit:
            df = df.sample(n=limit, random_state=42)

        df['source'] = f"hf:{dataset_name}"
        return df

    def _standardize_hf_columns(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Приведение колонок HF к стандартному виду."""
        if 'gazeta' in dataset_name.lower():
            column_mapping = {
                'title': 'title',
                'text': 'text',
                'summary': 'summary',
                'url': 'url',
                'date': 'published_at'
            }
        else:
            column_mapping = {
                'title': 'title',
                'text_markdown': 'text',
                'lead_markdown': 'summary',
                'url': 'url',
                'time_published': 'published_at'
            }

        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)

        if 'text' not in df.columns:
            if 'text_markdown' in df.columns:
                df['text'] = df['text_markdown']
            elif 'lead_markdown' in df.columns:
                df['text'] = df['lead_markdown']
            else:
                df['text'] = None

        if 'category' not in df.columns:
            if 'hubs' in df.columns:
                def _first(x):
                    if isinstance(x, list) and len(x) > 0:
                        return str(x[0]) if x[0] else None
                    return None
                df['category'] = df['hubs'].apply(_first)
            elif 'tags' in df.columns:
                def _first_tag(x):
                    if isinstance(x, list) and len(x) > 0:
                        return str(x[0]) if x[0] else None
                    return None
                df['category'] = df['tags'].apply(_first_tag)
            else:
                df['category'] = None

        for col in ['title', 'text', 'summary', 'url', 'published_at', 'category']:
            if col not in df.columns:
                df[col] = None

        if 'published_at' in df.columns and df['published_at'].notna().any():
            if df['published_at'].dtype == 'int64' or (hasattr(df['published_at'].dtype, 'kind') and df['published_at'].dtype.kind == 'i'):
                df['published_at'] = pd.to_datetime(df['published_at'], unit='s', errors='coerce')
            elif df['published_at'].dtype == object or str(df['published_at'].dtype) == 'string':
                df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

        return df

    def _raw_path(self, *parts: str) -> str:
        return os.path.join(self._project_root, "data", "raw", *parts)

    def _pick_main_kaggle_csv(self, csv_files: List[str]) -> Optional[str]:
        """Если в архиве несколько CSV, берём самый крупный (не sample)."""
        if not csv_files:
            return None
        existing = [p for p in csv_files if os.path.isfile(p)]
        if not existing:
            return None
        existing.sort(key=lambda p: os.path.getsize(p), reverse=True)
        for p in existing:
            if os.path.getsize(p) > 50_000:
                return p
        return existing[0]

    def _load_kaggle_dataset(self, source: Dict) -> Optional[pd.DataFrame]:
        """Загрузка датасета из Kaggle (локальный файл или API)."""
        dataset_name = source.get('name', 'yutkin/corpus-of-russian-news-articles-from-lenta')
        limit = source.get('limit', 2000)

        print(f"   Загрузка {dataset_name}...")

        safe_name = dataset_name.replace('/', '_')
        local_dir = self._raw_path(f"kaggle_{safe_name}")
        local_csv = self._raw_path(f"kaggle_{safe_name}.csv")

        df = None

        if os.path.exists(local_csv):
            df = pd.read_csv(local_csv)
            print(f"   Kaggle: из файла {local_csv}")
        elif os.path.isdir(local_dir):
            csv_files = glob.glob(os.path.join(local_dir, "**/*.csv"), recursive=True)
            pick = self._pick_main_kaggle_csv(csv_files)
            if pick:
                df = pd.read_csv(pick)
                print(f"   Kaggle: из кэша {pick}")
        if df is None or df.empty:
            df = self._download_kaggle_dataset(dataset_name, local_dir)

        if df is None or df.empty:
            kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
            print(
                f"   Kaggle: не удалось получить данные. Проверьте:\n"
                f"   • файл {kaggle_json} (chmod 600), см. README;\n"
                f"   • на сайте датасета нажата кнопка Download / приняты условия;\n"
                f"   • сеть: архивы часто 100–600+ МБ — дождитесь конца скачивания."
            )
            return None

        if len(df) > limit:
            df = df.sample(n=limit, random_state=42)

        df = self._standardize_kaggle_columns(df, dataset_name)
        df['source'] = f"kaggle:{dataset_name}"

        return df

    def _download_kaggle_dataset(self, dataset_name: str, local_dir: str) -> Optional[pd.DataFrame]:
        """Скачивание датасета через Kaggle API."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            print("   Kaggle: установите пакет: pip install kaggle")
            return None

        kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.isfile(kaggle_json):
            print(
                f"   Kaggle: нет {kaggle_json}. Скопируйте туда kaggle.json из профиля Kaggle "
                f"(Account → API → Create New Token), chmod 600."
            )
            return None

        try:
            api = KaggleApi()
            api.authenticate()
            os.makedirs(local_dir, exist_ok=True)
            print(
                f"   Kaggle API: скачивание {dataset_name} (архив может быть большим, идёт прогресс)..."
            )
            api.dataset_download_files(dataset_name, path=local_dir, unzip=True, quiet=False)

            csv_files = glob.glob(os.path.join(local_dir, "**/*.csv"), recursive=True)
            pick = self._pick_main_kaggle_csv(csv_files)
            if pick:
                print(f"   Kaggle: прочитан CSV {pick}")
                return pd.read_csv(pick)
            print(f"   Kaggle: в {local_dir} не найдено ни одного .csv после распаковки.")
        except Exception as e:
            print(f"   Ошибка Kaggle API: {e}")
        return None

    def _standardize_kaggle_columns(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Приведение колонок Kaggle к стандартному виду (разные датасеты)."""
        mappings = {
            'yutkin': {'title': 'title', 'text': 'text', 'url': 'url', 'topic': 'category', 'date': 'published_at'},
            'vfomenko': {'title': 'title', 'text': 'text', 'url': 'url', 'date': 'published_at'},
            'vyhuholl': {'title': 'title', 'text': 'text', 'url': 'url', 'date': 'published_at'},
        }

        mapping = None
        for key in mappings:
            if key in dataset_name.lower():
                mapping = {k: v for k, v in mappings[key].items() if k in df.columns}
                break

        if not mapping:
            mapping = {k: v for k, v in mappings['yutkin'].items() if k in df.columns}

        df = df.rename(columns=mapping)

        if 'summary' not in df.columns and 'text' in df.columns:
            df['summary'] = df['text'].astype(str).str[:200]

        for col in ['title', 'text', 'summary', 'url', 'published_at', 'category']:
            if col not in df.columns:
                df[col] = None

        if 'published_at' in df.columns and df['published_at'].notna().any():
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

        return df

    def _feed_name_from_url(self, url: str) -> str:
        """Извлечь имя источника из URL."""
        for domain in ['lenta.ru', 'ria.ru', 'tass.ru', 'kommersant.ru', 'rbc.ru']:
            if domain in url:
                return domain.split('.')[0]
        try:
            return url.split('/')[2].replace('.', '_')[:20]
        except IndexError:
            return 'unknown'

    def _load_rss_data(self, source: Dict) -> Optional[pd.DataFrame]:
        """Загрузка данных из RSS-лент. Сохраняет в data/raw/parsed_rss/<feed_name>/."""
        try:
            import feedparser
            import requests
        except ImportError:
            print("Установите: pip install feedparser requests")
            return None

        feeds = source.get('feeds', ['https://lenta.ru/rss/news'])
        if isinstance(feeds, str):
            feeds = [feeds]

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        all_dfs = []
        for feed_url in feeds:
            feed_name = self._feed_name_from_url(feed_url)
            local_dir = f"data/raw/parsed_rss/{feed_name}"
            local_csv = os.path.join(local_dir, "data.csv")

            df_feed = None
            if os.path.exists(local_csv):
                df_feed = pd.read_csv(local_csv)
                print(f"   RSS {feed_name}: загружено из кэша ({len(df_feed)} записей)")

            if df_feed is None or df_feed.empty:
                print(f"   Парсинг RSS: {feed_url[:55]}...")
                try:
                    resp = requests.get(feed_url, headers=headers, timeout=15)
                    resp.raise_for_status()
                    feed = feedparser.parse(resp.content)
                    limit_per_feed = source.get('limit_per_feed', 300)
                    entries = []
                    for entry in feed.entries[:limit_per_feed]:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', entry.get('description', ''))
                        pub_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                        if pub_struct:
                            from time import mktime
                            published_at = datetime.fromtimestamp(mktime(pub_struct))
                        else:
                            published_at = None
                        category = None
                        tags = entry.get('tags', [])
                        if tags and isinstance(tags[0], dict):
                            category = tags[0].get('term', tags[0].get('label'))
                        entries.append({
                            'title': title, 'text': summary or title,
                            'summary': (summary[:500] if summary else None) or title[:200],
                            'url': link, 'published_at': published_at, 'category': category
                        })
                    df_feed = pd.DataFrame(entries)
                    df_feed['source'] = f"parsed_rss:{feed_name}"
                    os.makedirs(local_dir, exist_ok=True)
                    df_feed.to_csv(local_csv, index=False, encoding='utf-8')
                    print(f"   Сохранено в {local_dir}")
                except Exception as e:
                    print(f"   Ошибка RSS {feed_name}: {e}")
                    continue

            if df_feed is not None and not df_feed.empty:
                all_dfs.append(df_feed)

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True)

    def _site_name_from_url(self, url: str) -> str:
        """Извлечь имя сайта из URL."""
        return self._feed_name_from_url(url)

    def _load_html_data(self, source: Dict) -> Optional[pd.DataFrame]:
        """Парсинг новостного сайта через BeautifulSoup. Сохраняет в data/raw/parsed_html/<site_name>/."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            print("Установите: pip install requests beautifulsoup4")
            return None

        url = source.get('url', 'https://lenta.ru/')
        limit = source.get('limit', 200)
        site_name = self._site_name_from_url(url)
        local_dir = f"data/raw/parsed_html/{site_name}"
        local_csv = os.path.join(local_dir, "data.csv")

        if os.path.exists(local_csv):
            df = pd.read_csv(local_csv)
            print(f"   HTML {site_name}: загружено из кэша ({len(df)} записей)")
            if len(df) > limit:
                df = df.sample(n=limit, random_state=42)
            return df

        print(f"   Парсинг HTML: {url}...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')

            entries = []
            seen_urls = set()
            if 'lenta.ru' in url:
                for a in soup.find_all('a', href=True):
                    if len(entries) >= limit:
                        break
                    href = a.get('href', '')
                    if '/news/' not in href and '/articles/' not in href:
                        continue
                    if not href.startswith('http'):
                        href = 'https://lenta.ru' + href
                    if href in seen_urls:
                        continue
                    seen_urls.add(href)
                    title = a.get_text(strip=True)
                    title = re.sub(r'\d{1,2}:\d{2}$', '', title).strip()
                    if len(title) > 15:
                        entries.append({
                            'title': title[:300],
                            'text': title,
                            'summary': title[:200],
                            'url': href,
                            'published_at': None,
                            'category': None
                        })
            elif 'ria.ru' in url:
                for a in soup.find_all('a', href=True):
                    if len(entries) >= limit:
                        break
                    href = a.get('href', '')
                    if '/20' not in href or 'ria.ru' not in href or href in seen_urls:
                        continue
                    seen_urls.add(href)
                    title = a.get_text(strip=True)
                    if len(title) > 20:
                        entries.append({
                            'title': title[:300],
                            'text': title,
                            'summary': title[:200],
                            'url': href,
                            'published_at': None,
                            'category': None
                        })
            elif 'tass.ru' in url:
                for a in soup.find_all('a', href=True):
                    if len(entries) >= limit:
                        break
                    href = a.get('href', '')
                    if not href or (href.startswith('#') or href.startswith('mailto:')):
                        continue
                    if not href.startswith('http'):
                        href = 'https://tass.ru' + href
                    if 'tass.ru' not in href:
                        continue
                    if href in seen_urls:
                        continue
                    seen_urls.add(href)
                    title = a.get_text(strip=True)
                    if len(title) > 15:
                        entries.append({
                            'title': title[:300],
                            'text': title,
                            'summary': title[:200],
                            'url': href,
                            'published_at': None,
                            'category': None
                        })
            else:
                links = soup.find_all('a', href=True)
                for a in links:
                    if len(entries) >= limit:
                        break
                    href = a.get('href', '')
                    if href.startswith('/'):
                        base = url.rstrip('/').split('/')[0] + '//' + url.split('/')[2]
                        href = base + href
                    text = a.get_text(strip=True)
                    if len(text) > 20 and href not in seen_urls and ('news' in href or 'article' in href or 'ru' in href):
                        seen_urls.add(href)
                        entries.append({
                            'title': text[:200],
                            'text': text,
                            'summary': text[:200],
                            'url': href,
                            'published_at': None,
                            'category': None
                        })

            if not entries:
                return None

            df = pd.DataFrame(entries)
            df['source'] = f"parsed_html:{site_name}"
            os.makedirs(local_dir, exist_ok=True)
            df.to_csv(local_csv, index=False, encoding='utf-8')
            print(f"   Сохранено в {local_dir}")
            return df

        except Exception as e:
            print(f"   Ошибка HTML парсинга: {e}")
            return None

    def _merge_sources(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Объединение данных с унификацией схемы."""
        common_columns = [
            'title', 'text', 'summary', 'url', 'published_at', 'category', 'source'
        ]

        unified_frames = []
        for df in dataframes:
            available_cols = [col for col in common_columns if col in df.columns]
            df_subset = df[available_cols].copy()

            for col in common_columns:
                if col not in df_subset.columns:
                    df_subset[col] = None

            unified_frames.append(df_subset[common_columns])

        return pd.concat(unified_frames, ignore_index=True)

    def save(self, df: pd.DataFrame, path: str = 'data/raw/unified_news.csv'):
        """Сохранение датасета в CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"Датасет сохранен: {path}")

    def get_stats(self) -> Dict:
        return self.collection_stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NewsCollectionAgent для новостей и медиа')
    parser.add_argument('--config', default='config.yaml', help='Путь к конфигу')
    parser.add_argument('--output', default='data/raw/unified_news.csv', help='Путь для сохранения')

    args = parser.parse_args()

    agent = DataCollectionAgent(config_path=args.config)
    df = agent.run()
    agent.save(df, args.output)
