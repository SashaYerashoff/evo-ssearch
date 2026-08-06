import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

export type UiLanguage = 'en' | 'lv'

const STORAGE_KEY = 'eva.ui.language.v1'

const EN = {
  'nav.home': 'Home',
  'nav.archive': 'Archive',
  'nav.summaries': 'Stream summaries',
  'nav.summariesShort': 'Summaries',
  'nav.probes': 'Probes',
  'nav.agent': 'Agent',
  'nav.settings': 'Settings',
  'nav.navigation': 'Navigation',
  'nav.logout': 'Log out',
  'nav.menu': 'MENU',
  'nav.openMenu': 'Open menu',
  'nav.closeMenu': 'Close menu',
  'appearance.title': 'Appearance',
  'appearance.open': 'Theme and appearance',
  'appearance.language': 'Interface language',
  'appearance.languageHelp': 'Changes EVA controls only. Evidence, summaries, reports, channel names, and agent conversations are not translated.',
  'appearance.english': 'English',
  'appearance.latvian': 'Latviešu',
  'auth.command': 'Command console · sign in',
  'auth.username': 'Username',
  'auth.password': 'Password',
  'auth.signIn': 'Sign in',
  'auth.signingIn': 'Signing in…',
  'auth.failed': 'Sign in failed',
  'status.checking': 'checking',
  'status.connected': 'connected',
  'status.stale': 'signal stale',
  'status.offline': 'offline',
  'status.luxriot': 'Luxriot',
  'status.channel': 'channel',
  'status.channels': 'channels',
  'status.agent': 'Agent',
  'status.idle': 'idle',
  'status.working': 'working',
  'status.capturing': 'capturing',
  'status.probe': 'configured probe',
  'status.probes': 'configured probes',
  'video.review': 'Stream review',
  'video.settings': 'Stream settings',
  'video.source': 'Stream source',
  'video.channel': 'Channel',
  'video.sampling': 'Sampling and batching',
  'video.batch': 'Batch',
  'video.every': 'Every (s)',
  'video.inference': 'Inference',
  'video.liveModel': 'Live model',
  'video.runtime': 'Runtime',
  'video.applyChanges': 'Apply changes',
  'video.stop': 'Stop',
  'video.start': 'Start summaries',
  'video.flush': 'Flush',
  'video.prompts': 'Prompts & alerts',
  'video.discard': 'Discard draft',
  'video.period': 'Period',
  'video.resolution': 'Resolution',
  'video.refresh': 'Refresh',
  'video.liveOn': 'Live on',
  'video.liveOff': 'Live off',
  'video.collapseAll': 'Collapse all',
  'video.expandAll': 'Expand all',
  'video.openPreview': 'Open preview',
  'video.editSettings': 'Edit settings',
  'video.from': 'From',
  'video.to': 'To',
  'period.live': 'Live',
  'period.today': 'Today',
  'period.yesterday': 'Yesterday',
  'period.dayBefore': 'Day before yesterday',
  'period.last7d': 'Last 7 days',
  'period.last30d': 'Last 30 days',
  'period.custom': 'Custom range…',
  'resolution.auto': 'Auto',
  'resolution.observations': 'Observations',
  'resolution.l1': '15 minute summaries',
  'resolution.l2': '1 hour summaries',
  'resolution.l3': '8 hour summaries',
  'common.copy': 'Copy',
  'common.export': 'Export',
  'common.bookmark': 'Bookmark',
  'common.reportIncident': 'Report incident',
  'common.saving': 'Saving…',
  'common.bookmarked': 'Bookmarked',
  'common.retry': 'Retry bookmark',
  'video.summaries': 'Stream summaries',
  'video.noChannel': 'No channel',
  'video.noSummaries': 'No summaries yet for this channel.',
  'incident.review': 'Incident review',
  'incident.tabSummary': 'Active · needs review · history',
  'incident.tabHelp': 'Review concurrent incident lifecycles without loading their full evidence until opened.',
  'incident.reviewHelp': 'Current focus, unresolved cases, and closed incident history across selected streams.',
  'incident.active': 'Active',
  'incident.needsReview': 'Needs review',
  'incident.history': 'History',
  'incident.allChannels': 'All channels',
  'incident.last24h': 'Last 24 hours',
  'incident.allTime': 'All time',
  'incident.loading': 'Loading incident ledger…',
  'incident.empty': 'No incidents in this queue for the selected scope.',
} as const

type TranslationKey = keyof typeof EN

const LV: Record<TranslationKey, string> = {
  'nav.home': 'Sākums',
  'nav.archive': 'Arhīvs',
  'nav.summaries': 'Straumju kopsavilkumi',
  'nav.summariesShort': 'Kopsavilkumi',
  'nav.probes': 'Pārbaudes',
  'nav.agent': 'Aģents',
  'nav.settings': 'Iestatījumi',
  'nav.navigation': 'Navigācija',
  'nav.logout': 'Izrakstīties',
  'nav.menu': 'IZVĒLNE',
  'nav.openMenu': 'Atvērt izvēlni',
  'nav.closeMenu': 'Aizvērt izvēlni',
  'appearance.title': 'Izskats',
  'appearance.open': 'Tēma un izskats',
  'appearance.language': 'Saskarnes valoda',
  'appearance.languageHelp': 'Maina tikai EVA vadības elementus. Pierādījumi, kopsavilkumi, pārskati, kanālu nosaukumi un aģenta sarunas netiek tulkotas.',
  'appearance.english': 'English',
  'appearance.latvian': 'Latviešu',
  'auth.command': 'Vadības konsole · pierakstīšanās',
  'auth.username': 'Lietotājvārds',
  'auth.password': 'Parole',
  'auth.signIn': 'Pierakstīties',
  'auth.signingIn': 'Pierakstās…',
  'auth.failed': 'Neizdevās pierakstīties',
  'status.checking': 'pārbauda',
  'status.connected': 'savienots',
  'status.stale': 'signāls novecojis',
  'status.offline': 'bezsaistē',
  'status.luxriot': 'Luxriot',
  'status.channel': 'kanāls',
  'status.channels': 'kanāli',
  'status.agent': 'Aģents',
  'status.idle': 'gaida',
  'status.working': 'strādā',
  'status.capturing': 'tver',
  'status.probe': 'konfigurēta pārbaude',
  'status.probes': 'konfigurētas pārbaudes',
  'video.review': 'Straumju pārskats',
  'video.settings': 'Straumju iestatījumi',
  'video.source': 'Straumes avots',
  'video.channel': 'Kanāls',
  'video.sampling': 'Kadru atlase un paketes',
  'video.batch': 'Pakete',
  'video.every': 'Ik pēc (s)',
  'video.inference': 'Modeļa izpilde',
  'video.liveModel': 'Aktīvais modelis',
  'video.runtime': 'Darbība',
  'video.applyChanges': 'Lietot izmaiņas',
  'video.stop': 'Apturēt',
  'video.start': 'Sākt kopsavilkumus',
  'video.flush': 'Nosūtīt paketi',
  'video.prompts': 'Uzvednes un trauksmes',
  'video.discard': 'Atmest melnrakstu',
  'video.period': 'Periods',
  'video.resolution': 'Detalizācija',
  'video.refresh': 'Atjaunot',
  'video.liveOn': 'Tiešraide ieslēgta',
  'video.liveOff': 'Tiešraide izslēgta',
  'video.collapseAll': 'Sakļaut visu',
  'video.expandAll': 'Izvērst visu',
  'video.openPreview': 'Atvērt priekšskatījumu',
  'video.editSettings': 'Rediģēt iestatījumus',
  'video.from': 'No',
  'video.to': 'Līdz',
  'period.live': 'Tiešraide',
  'period.today': 'Šodien',
  'period.yesterday': 'Vakar',
  'period.dayBefore': 'Aizvakar',
  'period.last7d': 'Pēdējās 7 dienas',
  'period.last30d': 'Pēdējās 30 dienas',
  'period.custom': 'Izvēlēts periods…',
  'resolution.auto': 'Automātiski',
  'resolution.observations': 'Novērojumi',
  'resolution.l1': '15 minūšu kopsavilkumi',
  'resolution.l2': '1 stundas kopsavilkumi',
  'resolution.l3': '8 stundu kopsavilkumi',
  'common.copy': 'Kopēt',
  'common.export': 'Eksportēt',
  'common.bookmark': 'Grāmatzīme',
  'common.reportIncident': 'Ziņot par incidentu',
  'common.saving': 'Saglabā…',
  'common.bookmarked': 'Saglabāts',
  'common.retry': 'Mēģināt vēlreiz',
  'video.summaries': 'Straumju kopsavilkumi',
  'video.noChannel': 'Kanāls nav izvēlēts',
  'video.noSummaries': 'Šim kanālam vēl nav kopsavilkumu.',
  'incident.review': 'Incidentu pārskats',
  'incident.tabSummary': 'Aktīvi · jāpārskata · vēsture',
  'incident.tabHelp': 'Pārskatiet vienlaicīgu incidentu dzīves ciklus, pilnos pierādījumus ielādējot tikai pēc atvēršanas.',
  'incident.reviewHelp': 'Aktīvais fokuss, neatrisinātie gadījumi un slēgto incidentu vēsture izvēlētajās straumēs.',
  'incident.active': 'Aktīvi',
  'incident.needsReview': 'Jāpārskata',
  'incident.history': 'Vēsture',
  'incident.allChannels': 'Visi kanāli',
  'incident.last24h': 'Pēdējās 24 stundas',
  'incident.allTime': 'Viss periods',
  'incident.loading': 'Ielādē incidentu žurnālu…',
  'incident.empty': 'Šajā rindā izvēlētajam tvērumam nav incidentu.',
}

interface I18nContextValue {
  language: UiLanguage
  locale: 'en-GB' | 'lv-LV'
  setLanguage: (language: UiLanguage) => void
  t: (key: TranslationKey) => string
}

const I18nContext = createContext<I18nContextValue | null>(null)

function initialLanguage(): UiLanguage {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === 'lv' ? 'lv' : 'en'
  } catch {
    return 'en'
  }
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<UiLanguage>(initialLanguage)
  const setLanguage = useCallback((next: UiLanguage) => {
    setLanguageState(next)
    try { window.localStorage.setItem(STORAGE_KEY, next) } catch { /* optional */ }
  }, [])
  useEffect(() => {
    document.documentElement.lang = language
  }, [language])
  const t = useCallback((key: TranslationKey) => (
    language === 'lv' ? LV[key] : EN[key]
  ), [language])
  const value = useMemo<I18nContextValue>(() => ({
    language,
    locale: language === 'lv' ? 'lv-LV' : 'en-GB',
    setLanguage,
    t,
  }), [language, setLanguage, t])
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext)
  if (!context) throw new Error('useI18n must be used inside I18nProvider')
  return context
}

export type { TranslationKey }
