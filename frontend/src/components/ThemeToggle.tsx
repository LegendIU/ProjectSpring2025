import { useEffect, useState } from 'react';

type ThemeMode = 'light' | 'dark';

const THEME_STORAGE_KEY = 'k3project-theme';

const getSystemTheme = (): ThemeMode => {
  if (typeof window === 'undefined' || !window.matchMedia) {
    return 'dark';
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

const ThemeToggle = () => {
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === 'undefined') {
      return 'dark';
    }
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null;
    return saved ?? getSystemTheme();
  });

  useEffect(() => {
    const root = document.body;
    root.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }

    const stored = window.localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null;
    const initialTheme = stored ?? getSystemTheme();
    setTheme(initialTheme);
  }, []);

  const toggleTheme = () => {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'));
  };

  return (
    <button className="theme-toggle" type="button" onClick={toggleTheme} aria-label="Переключить тему">
      <span className="theme-toggle__label">{theme === 'dark' ? 'Тёмная' : 'Светлая'} тема</span>
    </button>
  );
};

export default ThemeToggle;
