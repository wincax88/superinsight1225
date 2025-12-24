// i18next configuration
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import zhCommon from './zh/common.json';
import zhAuth from './zh/auth.json';
import zhDashboard from './zh/dashboard.json';
import enCommon from './en/common.json';
import enAuth from './en/auth.json';
import enDashboard from './en/dashboard.json';

const resources = {
  zh: {
    common: zhCommon,
    auth: zhAuth,
    dashboard: zhDashboard,
  },
  en: {
    common: enCommon,
    auth: enAuth,
    dashboard: enDashboard,
  },
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'zh',
    defaultNS: 'common',
    ns: ['common', 'auth', 'dashboard'],
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
    },
  });

export default i18n;
